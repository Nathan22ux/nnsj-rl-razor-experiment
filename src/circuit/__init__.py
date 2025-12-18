"""
Circuit Analysis Module for RL's Razor Replication

This module provides tools for:
1. Circuit Discovery via Path Patching (Equation 2)
2. DCM - Desiderata-based Component Masking (Equation 3)
3. Faithfulness Metrics (Equation 4)
4. Cross-Model Activation Patching - CMAP (Equation 5)
5. Circuit-Aware Regularization (Equation 6)
"""

from .discovery import (
    CircuitDiscovery,
    CrossModelCircuitAnalysis,
    DCMAnalysis,
    CircuitScore,
    DCMResult,
    create_counterfactual_examples_math,
    create_counterfactual_examples,
    save_circuit_results,
)

from .checkpoint_loader import (
    setup_circuit_analysis_models,
    load_your_checkpoint,
    find_best_checkpoint,
)

from .regularization import (
    CircuitAwareRegularizer,
    CircuitAwareTrainer,
    VulnerableHead,
    train_sft_with_circuit_regularization,
    load_vulnerable_heads_from_analysis,
)

__all__ = [
    # Discovery
    'CircuitDiscovery',
    'CrossModelCircuitAnalysis',
    'DCMAnalysis',
    'CircuitScore',
    'DCMResult',
    'create_counterfactual_examples_math',
    'create_counterfactual_examples',
    'save_circuit_results',

    # Checkpoint loading
    'setup_circuit_analysis_models',
    'load_your_checkpoint',
    'find_best_checkpoint',

    # Regularization
    'CircuitAwareRegularizer',
    'CircuitAwareTrainer',
    'VulnerableHead',
    'train_sft_with_circuit_regularization',
    'load_vulnerable_heads_from_analysis',
]
