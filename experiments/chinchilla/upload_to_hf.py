#!/usr/bin/env python3
"""
Sweep local training result directories and upload each run to its own
Hugging Face model repo.

Typical layout on the training machine:

    experiments/chinchilla/results/<run_name>/
        loss_log.csv
        loss_log_*.csv          # optional variants
        checkpoints/
            final.pt
            step_XXXXXXXX.pt
            early_stop_*.pt

Creates one repo per run under the FAIRC org by default
(https://huggingface.co/FAIRC):

    FAIRC/token-averaging-{tag}-{run_name}

where ``tag`` is empty for the canonical ``results/`` tree, and otherwise
the results-root name with the ``results_`` prefix stripped
(e.g. ``matched_mean``, ``update_control``).

Usage on the training machine (after ``huggingface-cli login`` or
``export HF_TOKEN=hf_...``):

    # Preview (only runs that have checkpoints/*.pt)
    python experiments/chinchilla/upload_to_hf.py --dry-run

    # Upload those runs to FAIRC (skips files already on the Hub)
    python experiments/chinchilla/upload_to_hf.py

    # Only the 500M / 1B ladder
    python experiments/chinchilla/upload_to_hf.py --only model1_500m avg_500m_k2 model1_1b avg_1b_k2

    # Re-upload everything even if already present
    python experiments/chinchilla/upload_to_hf.py --force

    # Also include loss-log-only dirs (no checkpoints)
    python experiments/chinchilla/upload_to_hf.py --include-logs-only

Requires:  pip install -U huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ROOTS = (
    "experiments/chinchilla/results",
    "experiments/chinchilla/results_matched_mean",
    "experiments/chinchilla/results_update_control",
)

# Files / patterns we always want in the repo. Checkpoints go under
# checkpoints/ as-is; we do not rewrite the .pt contents.
UPLOAD_GLOBS = (
    "loss_log*.csv",
    "checkpoints/*.pt",
    "checkpoints/*.bin",
    "eval*.json",
    "config.json",
    "README.md",
)

REPO_PREFIX = "token-averaging"
DEFAULT_NAMESPACE = "FAIRC"  # https://huggingface.co/FAIRC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tag_for_root(root: Path) -> str:
    """``results`` → ``""``; ``results_matched_mean`` → ``matched_mean``."""
    name = root.name
    if name == "results":
        return ""
    if name.startswith("results_"):
        return name[len("results_") :]
    return name


def _repo_id(namespace: str, tag: str, run_name: str) -> str:
    parts = [REPO_PREFIX]
    if tag:
        parts.append(tag)
    parts.append(run_name)
    return f"{namespace}/{'-'.join(parts)}"


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def _iter_upload_files(run_dir: Path) -> list[Path]:
    """Collect every file under ``run_dir`` that matches UPLOAD_GLOBS."""
    found: set[Path] = set()
    for pattern in UPLOAD_GLOBS:
        found.update(p for p in run_dir.glob(pattern) if p.is_file())
    # Also pick up any nested loss logs / json that sit at the run root
    # but were missed by the globs above (defensive).
    for p in run_dir.iterdir():
        if p.is_file() and (
            p.name.startswith("loss_log")
            or p.suffix in {".json", ".md"}
            or p.name == "config.json"
        ):
            found.add(p)
    return sorted(found)


def _list_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    return sorted(p for p in ckpt_dir.glob("*.pt") if p.is_file())


def _run_has_checkpoints(run_dir: Path) -> bool:
    return bool(_list_checkpoints(run_dir))


def _cfg_dict(run_name: str) -> Optional[dict[str, Any]]:
    try:
        from experiments.chinchilla.model_configs import MODEL_CONFIGS

        cfg = MODEL_CONFIGS.get(run_name)
        if cfg is None:
            return None
        d = asdict(cfg) if is_dataclass(cfg) else dict(cfg.__dict__)
        # Drop non-JSON-serialisable / plot-only fields that aren't needed
        # to rebuild the model.
        d["n_params_approx"] = getattr(cfg, "n_params_approx", None)
        return d
    except Exception as exc:  # noqa: BLE001 — best-effort metadata
        print(f"  [warn] could not load config for {run_name}: {exc}", flush=True)
        return None


def _write_sidecar(run_dir: Path, run_name: str, tag: str, repo_id: str) -> None:
    """Write config.json + README.md into the run dir (uploaded with the rest)."""
    cfg = _cfg_dict(run_name)
    meta = {
        "run_name": run_name,
        "results_tag": tag or "canonical",
        "hf_repo_id": repo_id,
        "checkpoint_format": {
            "keys": ["step", "tokens_seen", "cumulative_flops", "model", "optimizer", "scheduler"],
            "note": (
                "Raw torch.save dict from experiments/chinchilla/train.py. "
                "Load with torch.load(..., map_location='cpu', weights_only=False) "
                "and take state['model']. Not a transformers AutoModel checkpoint."
            ),
        },
        "model_config": cfg,
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    ckpt_names = [p.name for p in _list_checkpoints(run_dir)]
    loss_names = sorted(p.name for p in run_dir.glob("loss_log*.csv"))

    lines = [
        f"# {repo_id}",
        "",
        "Checkpoint dump from the **token averaging** research project.",
        "",
        f"- **run name:** `{run_name}`",
        f"- **results tree:** `{tag or 'results'}`",
        "",
        "## Contents",
        "",
        "### Loss logs",
        "",
    ]
    if loss_names:
        lines += [f"- `{n}`" for n in loss_names]
    else:
        lines.append("- _(none)_")
    lines += ["", "### Checkpoints", ""]
    if ckpt_names:
        lines += [f"- `checkpoints/{n}`" for n in ckpt_names]
    else:
        lines.append("- _(none — loss logs only)_")
    lines += [
        "",
        "## Loading a checkpoint",
        "",
        "```python",
        "import torch",
        "from huggingface_hub import hf_hub_download",
        "",
        f"path = hf_hub_download({repo_id!r}, 'checkpoints/final.pt')",
        "state = torch.load(path, map_location='cpu', weights_only=False)",
        "model.load_state_dict(state['model'])  # your OLMAveraged / OLMTransformerBody",
        "print(state['step'], state['tokens_seen'], state['cumulative_flops'])",
        "```",
        "",
        "These are **not** Hugging Face `transformers` weights. Rebuild the",
        "architecture from `config.json` → `model_config` (or from",
        "`experiments/chinchilla/model_configs.py` in the source repo) and load",
        "the raw `state_dict`.",
        "",
    ]
    if cfg:
        lines += [
            "## Architecture",
            "",
            "```json",
            json.dumps(
                {
                    k: cfg[k]
                    for k in (
                        "d_model",
                        "n_heads",
                        "n_layers",
                        "context_len",
                        "averaging_k",
                        "tie_embeddings",
                        "method_name",
                        "lr",
                        "warmup_steps",
                        "target_tokens",
                        "n_params_approx",
                    )
                    if k in cfg and cfg[k] is not None
                },
                indent=2,
            ),
            "```",
            "",
        ]
    (run_dir / "README.md").write_text("\n".join(lines))


def discover_runs(
    roots: Iterable[Path],
    only: Optional[set[str]] = None,
    *,
    require_checkpoints: bool = True,
) -> list[tuple[Path, str, str]]:
    """Return ``[(run_dir, tag, run_name), ...]``.

    By default only runs that have at least one ``checkpoints/*.pt`` are
    included — loss-log-only directories do not get a repo.
    """
    out: list[tuple[Path, str, str]] = []
    for root in roots:
        if not root.is_dir():
            print(f"[skip] results root missing: {root}", flush=True)
            continue
        tag = _tag_for_root(root)
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            run_name = child.name
            if only is not None and run_name not in only:
                continue
            ckpts = _list_checkpoints(child)
            if require_checkpoints:
                if not ckpts:
                    print(f"[skip] no checkpoints/: {child}", flush=True)
                    continue
            elif not _iter_upload_files(child) and not ckpts:
                print(f"[skip] empty: {child}", flush=True)
                continue
            out.append((child, tag, run_name))
            if ckpts:
                try:
                    rel = child.relative_to(_ROOT)
                except ValueError:
                    rel = child
                names = ", ".join(p.name for p in ckpts)
                print(
                    f"[found] {rel} → {len(ckpts)} ckpt(s): {names}",
                    flush=True,
                )
    return out


def _remote_paths(api, repo_id: str) -> set[str]:
    """Return paths already in the remote repo, or empty set if repo missing."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except RepositoryNotFoundError:
        return set()
    except Exception as exc:  # noqa: BLE001
        # Older hub versions / private-repo edge cases.
        try:
            from huggingface_hub.utils import EntryNotFoundError  # noqa: F401
        except ImportError:
            pass
        print(f"  [warn] list_repo_files({repo_id}) failed: {exc}", flush=True)
        return set()


def ensure_repo(api, repo_id: str, private: bool, exist_ok: bool = True) -> None:
    from huggingface_hub.utils import HfHubHTTPError

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=exist_ok,
        )
    except HfHubHTTPError as exc:
        # Race / already exists with different visibility — surface clearly.
        raise RuntimeError(f"create_repo({repo_id}) failed: {exc}") from exc


ALLOW_PATTERNS = [
    "loss_log*.csv",
    "checkpoints/*.pt",
    "checkpoints/*.bin",
    "eval*.json",
    "config.json",
    "README.md",
]


def upload_run(
    api,
    run_dir: Path,
    repo_id: str,
    *,
    private: bool,
    dry_run: bool,
    large_folder: bool,
    skip_existing: bool = True,
    revision: str = "main",
) -> None:
    files = _iter_upload_files(run_dir)
    remote: set[str] = set()
    if skip_existing and not dry_run and api is not None:
        remote = _remote_paths(api, repo_id)
    elif skip_existing and dry_run and api is not None:
        remote = _remote_paths(api, repo_id)

    to_upload: list[Path] = []
    skipped: list[Path] = []
    for p in files:
        rel = str(p.relative_to(run_dir)).replace("\\", "/")
        # Always refresh small sidecar metadata so README/config stay current.
        if skip_existing and remote and rel in remote and rel not in (
            "README.md",
            "config.json",
        ):
            skipped.append(p)
        else:
            to_upload.append(p)

    total = sum(p.stat().st_size for p in to_upload)
    ckpts = _list_checkpoints(run_dir)
    print(
        f"\n=== {repo_id} ===\n"
        f"  local: {run_dir}\n"
        f"  checkpoints on disk: {len(ckpts)} "
        f"({', '.join(p.name for p in ckpts) or 'none'})\n"
        f"  upload: {len(to_upload)} file(s)  ({_human_bytes(total)})"
        + (f"  |  skip existing: {len(skipped)}" if skipped else ""),
        flush=True,
    )
    for p in to_upload:
        print(f"    + {p.relative_to(run_dir)}  ({_human_bytes(p.stat().st_size)})", flush=True)
    for p in skipped:
        print(f"    = {p.relative_to(run_dir)}  (already on Hub)", flush=True)

    if not to_upload:
        print("  nothing new to upload", flush=True)
        return

    if dry_run:
        print("  [dry-run] skip create/upload", flush=True)
        return

    ensure_repo(api, repo_id, private=private)
    print(f"  repo ready (private={private})", flush=True)

    # Build allow_patterns from the concrete relative paths we decided to
    # push, so skip-existing is enforced even for upload_folder / large_folder.
    allow = sorted({str(p.relative_to(run_dir)).replace("\\", "/") for p in to_upload})

    commit_msg = f"Upload {run_dir.name}: {len(to_upload)} new file(s)"
    t0 = time.time()

    if large_folder or total > 5 * 1024**3:
        print("  uploading via upload_large_folder …", flush=True)
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(run_dir),
            repo_type="model",
            revision=revision,
            allow_patterns=allow,
            ignore_patterns=["**/*.tmp", "**/.*", "**/*~"],
        )
    else:
        print("  uploading via upload_folder …", flush=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(run_dir),
            repo_type="model",
            revision=revision,
            commit_message=commit_msg,
            allow_patterns=allow,
            ignore_patterns=["**/*.tmp", "**/.*", "**/*~"],
        )

    dt = time.time() - t0
    print(f"  done in {dt/60:.1f} min → https://huggingface.co/{repo_id}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create one HF repo per training run that has checkpoints, "
            "and upload only files not already on the Hub."
        ),
    )
    p.add_argument(
        "--results-root",
        action="append",
        dest="results_roots",
        default=None,
        help=(
            "Results directory to sweep (repeatable). "
            f"Default: {', '.join(DEFAULT_ROOTS)}"
        ),
    )
    p.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"HF user or org. Default: {DEFAULT_NAMESPACE} (https://huggingface.co/FAIRC).",
    )
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only upload these run names (e.g. model1_500m avg_500m_k2).",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Create public repos (default: private).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List repos/files only; do not create or upload.",
    )
    p.add_argument(
        "--large-folder",
        action="store_true",
        help="Force resumable upload_large_folder for every run (default: auto when >5 GB).",
    )
    p.add_argument(
        "--include-old",
        action="store_true",
        help="Also sweep experiments/chinchilla/results_old.",
    )
    p.add_argument(
        "--include-logs-only",
        action="store_true",
        help="Also upload runs that have loss logs but no checkpoints/ (default: skip them).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-upload files even if the same path already exists on the Hub.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token. Default: $HF_TOKEN / cached login.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    roots_raw = args.results_roots or list(DEFAULT_ROOTS)
    if args.include_old:
        roots_raw.append("experiments/chinchilla/results_old")
    roots = [(_ROOT / r if not Path(r).is_absolute() else Path(r)) for r in roots_raw]

    only = set(args.only) if args.only else None
    require_ckpts = not args.include_logs_only
    skip_existing = not args.force

    print(
        f"Mode: require_checkpoints={require_ckpts}  "
        f"skip_existing={skip_existing}",
        flush=True,
    )
    runs = discover_runs(roots, only=only, require_checkpoints=require_ckpts)
    if not runs:
        print(
            "No runs with checkpoints found to upload.\n"
            "  (pass --include-logs-only to also upload loss-log-only dirs)",
            flush=True,
        )
        return 1

    print(f"Discovered {len(runs)} run(s) with checkpoints across {len(roots)} root(s).", flush=True)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    namespace = args.namespace  # default: FAIRC
    api = None

    print(f"Target namespace: {namespace}", flush=True)

    # Need the API whenever we upload, or when dry-running with skip-existing
    # so we can show which remote files would be skipped.
    need_api = (not args.dry_run) or skip_existing
    if need_api:
        try:
            from huggingface_hub import HfApi, whoami
        except ImportError:
            print(
                "ERROR: huggingface_hub is not installed.\n"
                "  pip install -U huggingface_hub",
                file=sys.stderr,
            )
            return 1
        api = HfApi(token=token)
        try:
            info = whoami(token=token)
            auth_name = info.get("name") or info.get("fullname") or "?"
            orgs = [o.get("name") for o in info.get("orgs", []) if isinstance(o, dict)]
            print(f"Authenticated as: {auth_name}", flush=True)
            if orgs:
                print(f"  orgs: {', '.join(orgs)}", flush=True)
            if namespace != auth_name and namespace not in orgs:
                print(
                    f"  [warn] '{namespace}' is not in your org list — "
                    "create_repo will fail if you lack write access.",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] whoami failed ({exc}); continuing anyway", flush=True)

    private = not args.public
    failures: list[str] = []

    for run_dir, tag, run_name in runs:
        repo_id = _repo_id(namespace, tag, run_name)
        try:
            _write_sidecar(run_dir, run_name, tag, repo_id)
            upload_run(
                api,
                run_dir,
                repo_id,
                private=private,
                dry_run=args.dry_run,
                large_folder=args.large_folder,
                skip_existing=skip_existing,
            )
        except Exception as exc:  # noqa: BLE001 — keep sweeping other runs
            print(f"  FAILED: {exc}", flush=True)
            failures.append(f"{repo_id}: {exc}")

    print("\n──────── summary ────────", flush=True)
    print(f"attempted: {len(runs)}", flush=True)
    print(f"failed:    {len(failures)}", flush=True)
    for f in failures:
        print(f"  - {f}", flush=True)
    if args.dry_run:
        print("(dry-run — nothing was uploaded)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
