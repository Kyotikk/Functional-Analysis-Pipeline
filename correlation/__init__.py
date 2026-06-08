"""Phase 4 — R4 clinical validation and ICF-inspired correlation analysis."""
from .merger    import load_and_merge
from .validator import validate, ValidationResult, CorrelationResult, results_to_dataframe
from .reporter  import write_outputs

__all__ = [
    "load_and_merge",
    "validate",
    "write_outputs",
    "ValidationResult",
    "CorrelationResult",
    "results_to_dataframe",
]
