"""
Utility modules for token averaging research.

Imports are lazy (PEP 562). Submodules like ``utils.averaging_methods`` need
only torch, but ``model_loader`` and ``data_loader`` pull in transformers and
datasets. Importing those eagerly here meant that any transformers/torch
version mismatch broke training and evaluation code that never touches them.
"""

import importlib

# public name -> submodule that defines it
_EXPORTS = {
    "load_pythia_model": ".model_loader",
    "load_wikitext103": ".data_loader",
    "get_data_iterator": ".data_loader",
    "extract_embeddings": ".embedding_extractor",
    "apply_averaging": ".embedding_extractor",
    "setup_plot_style": ".visualization",
    "save_figure": ".visualization",
    "setup_logging": ".runner_utils",
    "collect_embeddings": ".runner_utils",
    "run_analyses_for_averaged": ".runner_utils",
    "flatten_results_to_rows": ".runner_utils",
    "export_results_to_csv": ".runner_utils",
    "export_results_to_json": ".runner_utils",
    "create_summary_report": ".runner_utils",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value       # cache so later lookups skip this path
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
