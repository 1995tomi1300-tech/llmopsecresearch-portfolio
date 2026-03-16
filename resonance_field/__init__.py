from resonance_field.config import ExperimentConfig, Scenario, build_scenarios
from resonance_field.spiral_validation import (
    SpiralSignalVector,
    SpiralValidationConfig,
    SpiralValidationEngine,
    SpiralValidationResult,
    evaluate_report,
    load_report_candidates,
)
from resonance_field.spiral_bridge import (
    BRIDGE_VERSION,
    SpiralBridgeRecord,
    export_bridge_records,
    write_bridge_payload,
)

try:
    from resonance_field.experiment import run_experiments
except ModuleNotFoundError:
    run_experiments = None

__all__ = [
    "ExperimentConfig",
    "Scenario",
    "build_scenarios",
    "run_experiments",
    "SpiralSignalVector",
    "SpiralValidationConfig",
    "SpiralValidationEngine",
    "SpiralValidationResult",
    "evaluate_report",
    "load_report_candidates",
    "BRIDGE_VERSION",
    "SpiralBridgeRecord",
    "export_bridge_records",
    "write_bridge_payload",
]
