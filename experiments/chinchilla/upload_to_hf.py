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

    # Upload those runs to FAIRC (all checkpoints; skips files already on Hub)
    python experiments/chinchilla/upload_to_hf.py

    # If private storage quota is full, make repos public:
    python experiments/chinchilla/upload_to_hf.py --public

    # Broken LFS pointers: wipe repo and re-upload final.pt only first
    python experiments/chinchilla/upload_to_hf.py --public --recreate --final-only

    # Then (optional) push the rest of the step checkpoints into the clean repo
    python experiments/chinchilla/upload_to_hf.py --public

    # Only upload checkpoints/final.pt
    python experiments/chinchilla/upload_to_hf.py --public --final-only

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

UPLOAD_GLOBS_FINAL_ONLY = (
    "loss_log*.csv",
    "checkpoints/final.pt",
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


def _iter_upload_files(run_dir: Path, *, final_only: bool = False) -> list[Path]:
    """Collect files to upload. Default: loss logs + every ``checkpoints/*.pt``."""
    patterns = UPLOAD_GLOBS_FINAL_ONLY if final_only else UPLOAD_GLOBS
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in run_dir.glob(pattern) if p.is_file())
    for p in run_dir.iterdir():
        if p.is_file() and (
            p.name.startswith("loss_log")
            or p.suffix in {".json", ".md"}
            or p.name == "config.json"
        ):
            found.add(p)
    return sorted(found)


def _list_checkpoints(run_dir: Path, *, final_only: bool = False) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    if final_only:
        final = ckpt_dir / "final.pt"
        return [final] if final.is_file() else []
    return sorted(p for p in ckpt_dir.glob("*.pt") if p.is_file())


def _run_has_checkpoints(run_dir: Path, *, final_only: bool = False) -> bool:
    return bool(_list_checkpoints(run_dir, final_only=final_only))


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

    ckpt_names = [p.name for p in _list_checkpoints(run_dir, final_only=False)]
    # README lists every local checkpoint for transparency, even when we
    # only upload final.pt.
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
    final_only: bool = False,
) -> list[tuple[Path, str, str]]:
    """Return ``[(run_dir, tag, run_name), ...]``.

    By default only runs that have at least one ``checkpoints/*.pt`` are
    included. With ``final_only=True``, require ``checkpoints/final.pt``.
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
            ckpts = _list_checkpoints(child, final_only=final_only)
            if require_checkpoints:
                if not ckpts:
                    kind = "final.pt" if final_only else "checkpoints/*.pt"
                    print(f"[skip] no {kind}: {child}", flush=True)
                    continue
            elif not _iter_upload_files(child, final_only=final_only) and not ckpts:
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
                    f"[found] {rel} → {len(ckpts)} ckpt(s) to consider: {names}",
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
        raise RuntimeError(f"create_repo({repo_id}) failed: {exc}") from exc

    if not private:
        try:
            api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not set public: {exc}", flush=True)


def recreate_repo(api, repo_id: str, private: bool, *, dry_run: bool) -> None:
    """Delete the repo entirely and create a fresh empty one.

    Reliable fix when dangling LFS pointers block every subsequent commit
    (including path deletes via --repair).
    """
    from huggingface_hub.utils import RepositoryNotFoundError

    print(f"  recreate: deleting repo {repo_id} …", flush=True)
    if dry_run:
        print("  [dry-run] skip delete/create", flush=True)
        return
    try:
        api.delete_repo(repo_id=repo_id, repo_type="model")
        time.sleep(2)
    except RepositoryNotFoundError:
        print("  recreate: repo did not exist yet", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] delete_repo failed ({exc}); trying create anyway", flush=True)
    ensure_repo(api, repo_id, private=private, exist_ok=True)
    print(f"  recreate: fresh repo ready (private={private})", flush=True)


def _clear_local_upload_cache(run_dir: Path) -> None:
    """Drop resumable-upload metadata that can re-commit broken LFS pointers."""
    import shutil

    cache = run_dir / ".cache" / ".huggingface"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)
        print(f"  cleared local upload cache: {cache}", flush=True)


def _repair_remote_checkpoints(api, repo_id: str, *, dry_run: bool) -> None:
    """Delete remote checkpoints/* that may be dangling LFS pointers."""
    from huggingface_hub import CommitOperationDelete

    remote = _remote_paths(api, repo_id)
    bad = sorted(
        p
        for p in remote
        if p.startswith("checkpoints/") and (p.endswith(".pt") or p.endswith(".bin"))
    )
    if not bad:
        print("  repair: no remote checkpoints/ to delete", flush=True)
        return
    print(f"  repair: deleting {len(bad)} remote checkpoint path(s) …", flush=True)
    for p in bad:
        print(f"    - {p}", flush=True)
    if dry_run:
        print("  [dry-run] skip remote delete", flush=True)
        return
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[CommitOperationDelete(path_in_repo=p) for p in bad],
            commit_message="Repair: remove dangling LFS checkpoint pointers",
        )
    except Exception as exc:
        raise RuntimeError(
            f"repair delete failed ({exc}).\n"
            "  Repo LFS state is too broken for path deletes — re-run with --recreate"
        ) from exc


def _upload_files_single_commit(
    api,
    run_dir: Path,
    repo_id: str,
    files: list[Path],
    *,
    revision: str,
    commit_message: str,
) -> None:
    """Upload every file in one Hub commit (LFS-aware via CommitOperationAdd)."""
    from huggingface_hub import CommitOperationAdd

    ops = []
    for p in files:
        rel = str(p.relative_to(run_dir)).replace("\\", "/")
        size = _human_bytes(p.stat().st_size)
        print(f"    stage {rel} ({size})", flush=True)
        ops.append(
            CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p))
        )
    print(f"  committing {len(ops)} file(s) in a single commit …", flush=True)
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=ops,
        commit_message=commit_message,
        revision=revision,
    )


def upload_run(
    api,
    run_dir: Path,
    repo_id: str,
    *,
    private: bool,
    dry_run: bool,
    large_folder: bool,
    skip_existing: bool = True,
    final_only: bool = False,
    repair: bool = False,
    recreate: bool = False,
    revision: str = "main",
) -> None:
    _clear_local_upload_cache(run_dir)

    if recreate and api is not None:
        recreate_repo(api, repo_id, private=private, dry_run=dry_run)
        skip_existing = False
    elif repair and api is not None:
        if not dry_run:
            ensure_repo(api, repo_id, private=private)
        _repair_remote_checkpoints(api, repo_id, dry_run=dry_run)
        skip_existing = False

    files = _iter_upload_files(run_dir, final_only=final_only)
    remote: set[str] = set()
    if skip_existing and api is not None and not recreate:
        remote = _remote_paths(api, repo_id)

    to_upload: list[Path] = []
    skipped: list[Path] = []
    for p in files:
        rel = str(p.relative_to(run_dir)).replace("\\", "/")
        if skip_existing and remote and rel in remote and rel not in (
            "README.md",
            "config.json",
        ):
            skipped.append(p)
        else:
            to_upload.append(p)

    total = sum(p.stat().st_size for p in to_upload)
    ckpts = _list_checkpoints(run_dir, final_only=final_only)
    print(
        f"\n=== {repo_id} ===\n"
        f"  local: {run_dir}\n"
        f"  checkpoints selected: {len(ckpts)} "
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

    if not recreate:
        ensure_repo(api, repo_id, private=private)
    print(f"  repo ready (private={private})", flush=True)

    t0 = time.time()
    commit_msg = f"Upload {run_dir.name}: {len(to_upload)} file(s)"

    try:
        if large_folder:
            allow = sorted(
                {str(p.relative_to(run_dir)).replace("\\", "/") for p in to_upload}
            )
            print("  uploading via upload_large_folder …", flush=True)
            api.upload_large_folder(
                repo_id=repo_id,
                folder_path=str(run_dir),
                repo_type="model",
                revision=revision,
                allow_patterns=allow,
                ignore_patterns=["**/*.tmp", "**/.*", "**/*~", "**/.cache/**"],
            )
        else:
            # One create_commit for the whole run: Hub pre-uploads LFS blobs,
            # then records every path in a single commit.
            print("  uploading in a single commit …", flush=True)
            _upload_files_single_commit(
                api,
                run_dir,
                repo_id,
                to_upload,
                revision=revision,
                commit_message=commit_msg,
            )
    except Exception as exc:
        msg = str(exc)
        if "Private repository storage limit" in msg or "storage limit" in msg.lower():
            raise RuntimeError(
                f"{exc}\n\n"
                "HF private storage quota is full. Re-run with --public."
            ) from exc
        if "LFS pointer" in msg:
            raise RuntimeError(
                f"{exc}\n\n"
                "Broken LFS pointers on the Hub.\n"
                "  Wipe and re-upload cleanly:\n"
                "    python experiments/chinchilla/upload_to_hf.py --public --recreate --final-only"
            ) from exc
        raise

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
        help="Force resumable upload_large_folder (not recommended if LFS was broken).",
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
        "--final-only",
        action="store_true",
        help="Upload only checkpoints/final.pt (default: every checkpoints/*.pt).",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete each target HF repo and recreate it empty before uploading. "
            "Use this when dangling LFS pointers make every commit fail."
        ),
    )
    p.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Delete remote checkpoints/* before uploading. Prefer --recreate if "
            "this also fails with an LFS pointer error."
        ),
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
    skip_existing = not args.force and not args.recreate
    final_only = args.final_only

    print(
        f"Mode: final_only={final_only}  require_checkpoints={require_ckpts}  "
        f"skip_existing={skip_existing}  private={not args.public}  "
        f"repair={args.repair}  recreate={args.recreate}",
        flush=True,
    )
    if not args.public:
        print(
            "Note: private HF storage is limited. If you hit the quota, re-run with --public.",
            flush=True,
        )
    runs = discover_runs(
        roots,
        only=only,
        require_checkpoints=require_ckpts,
        final_only=final_only,
    )
    if not runs:
        print(
            "No runs with checkpoints found to upload.\n"
            "  (pass --include-logs-only to also upload loss-log-only dirs)",
            flush=True,
        )
        return 1

    print(f"Discovered {len(runs)} run(s) with checkpoints across {len(roots)} root(s).", flush=True)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    namespace = args.namespace
    api = None

    print(f"Target namespace: {namespace}", flush=True)

    need_api = (not args.dry_run) or skip_existing or args.repair or args.recreate
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
                final_only=args.final_only,
                repair=args.repair,
                recreate=args.recreate,
            )
        except Exception as exc:  # noqa: BLE001
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
