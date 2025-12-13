"""Circuit Analysis Package"""

from .discovery import (
    CircuitDiscovery,
    CircuitScore,
    CrossModelCircuitAnalysis,
    create_counterfactual_examples_math,
)
from .visualization import (
    plot_circuit_overlap,
    plot_cmap_comparison,
    plot_vulnerable_circuits,
    plot_circuit_heatmap,
    generate_all_visualizations,
    print_circuit_summary
)
from .checkpoint_loader import (
    find_best_checkpoint,
    load_your_checkpoint,
    setup_circuit_analysis_models,
    load_models_for_circuit_analysis
)

__all__ = [
    'CircuitDiscovery',
    'CircuitScore',
    'CrossModelCircuitAnalysis',
    'create_counterfactual_examples_math',
    'plot_circuit_overlap',
    'plot_cmap_comparison',
    'plot_vulnerable_circuits',
    'plot_circuit_heatmap',
    'generate_all_visualizations',
    'print_circuit_summary',
    'find_best_checkpoint',
    'load_your_checkpoint',
    'setup_circuit_analysis_models',
    'load_models_for_circuit_analysis',
]

__version__ = '1.0.0'
