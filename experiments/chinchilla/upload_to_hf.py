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

    # Preview what would be uploaded
    python experiments/chinchilla/upload_to_hf.py --dry-run

    # Upload everything under the default roots (private repos)
    python experiments/chinchilla/upload_to_hf.py

    # Only the 500M / 1B ladder
    python experiments/chinchilla/upload_to_hf.py --only model1_500m avg_500m_k2 model1_1b avg_1b_k2

    # Public repos (still under FAIRC by default)
    python experiments/chinchilla/upload_to_hf.py --public

    # Different namespace
    python experiments/chinchilla/upload_to_hf.py --namespace my-user

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


def _run_has_content(run_dir: Path) -> bool:
    files = _iter_upload_files(run_dir)
    # A run with only a README we just wrote does not count as content.
    return any(
        p.name.startswith("loss_log")
        or p.suffix == ".pt"
        or p.name.startswith("eval")
        for p in files
    ) or (run_dir / "checkpoints").is_dir() and any(
        (run_dir / "checkpoints").glob("*.pt")
    )


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

    ckpt_dir = run_dir / "checkpoints"
    ckpt_names = sorted(p.name for p in ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
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
) -> list[tuple[Path, str, str]]:
    """Return ``[(run_dir, tag, run_name), ...]`` for every non-empty run."""
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
            if not _run_has_content(child):
                print(f"[skip] empty / no logs or ckpts: {child}", flush=True)
                continue
            out.append((child, tag, run_name))
    return out


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


def upload_run(
    api,
    run_dir: Path,
    repo_id: str,
    *,
    private: bool,
    dry_run: bool,
    large_folder: bool,
    revision: str = "main",
) -> None:
    files = _iter_upload_files(run_dir)
    total = sum(p.stat().st_size for p in files)
    print(
        f"\n=== {repo_id} ===\n"
        f"  local: {run_dir}\n"
        f"  files: {len(files)}  ({_human_bytes(total)})",
        flush=True,
    )
    for p in files:
        rel = p.relative_to(run_dir)
        print(f"    {rel}  ({_human_bytes(p.stat().st_size)})", flush=True)

    if dry_run:
        print("  [dry-run] skip create/upload", flush=True)
        return

    ensure_repo(api, repo_id, private=private)
    print(f"  repo ready (private={private})", flush=True)

    commit_msg = f"Upload {run_dir.name} checkpoints + loss logs"
    t0 = time.time()

    if large_folder or total > 5 * 1024**3:
        # Resumable multi-worker path — right choice for 500M / 1B dumps.
        print("  uploading via upload_large_folder …", flush=True)
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(run_dir),
            repo_type="model",
            revision=revision,
            # Only push what we care about; ignore anything else that
            # might have landed in the run dir (tmp, .pt.tmp, etc.).
            allow_patterns=[
                "loss_log*.csv",
                "checkpoints/*.pt",
                "checkpoints/*.bin",
                "eval*.json",
                "config.json",
                "README.md",
            ],
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
            allow_patterns=[
                "loss_log*.csv",
                "checkpoints/*.pt",
                "checkpoints/*.bin",
                "eval*.json",
                "config.json",
                "README.md",
            ],
            ignore_patterns=["**/*.tmp", "**/.*", "**/*~"],
        )

    dt = time.time() - t0
    print(f"  done in {dt/60:.1f} min → https://huggingface.co/{repo_id}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create one HF repo per training run and upload checkpoints + loss logs.",
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
    roots = [( _ROOT / r if not Path(r).is_absolute() else Path(r)) for r in roots_raw]

    only = set(args.only) if args.only else None
    runs = discover_runs(roots, only=only)
    if not runs:
        print("No runs found to upload.", flush=True)
        return 1

    print(f"Discovered {len(runs)} run(s) across {len(roots)} root(s).", flush=True)

    # Resolve namespace / API only when we need them (dry-run still needs
    # the namespace to print the intended repo ids).
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    namespace = args.namespace  # default: FAIRC
    api = None

    print(f"Target namespace: {namespace}", flush=True)

    if not args.dry_run:
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
