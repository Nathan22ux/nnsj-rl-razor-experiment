Root Level
main.py - Entry point for running experiments and training pipelines
run_circuit_analysis.py - Script to execute circuit discovery and analysis workflows
requirements.txt - Python dependencies for the project
README.md - Project documentation and folder structure overview

logger.py - Logging utility for tracking experiments and debugging

CONFIG.py - Central configuration file for hyperparameters and settings

load_data.py - Loads datasets from JSON/JSONL files
dataset_utils.py - Utilities for data preprocessing and batching

load_model.py - Loads pretrained language models
regularization.py - Implements regularization techniques during training

*training.py - Core training loop and optimization logic*
*experiment.py - Manages experiment runs and hyperparameter sweeps*

train_dr_grpo.py - GRPO training with Distribution Reweighting (DR) loss
train_sft_baseline.py - Supervised Fine-Tuning baseline
dr_loss.py - Distribution Reweighting loss implementation
reward.py - Reward signal calculation
rollout.py - Generates rollouts for RL training
advantages.py - Computes advantage estimates
eval_forgetting.py - Evaluates catastrophic forgetting
eval_kl_forward.py - Measures KL divergence between policies
eval_pareto.py - Analyzes Pareto frontier of performance vs KL
nu_loop.py - Hyperparameter tuning loop for RL coefficients
plot_rls_razor.py - Generates visualizations for RL results

evaluation.py - Computes metrics (accuracy, F1, etc.) on test sets

verify_setup.py - Validates environment and dependencies
verify_activation.py - Checks neuron activations are properly captured


visualization.py - General plotting utilities
visualize_circuits.py - Visualizes circuit structures and activations

# to check formats

Get-Content src/data/science/balancing_chemical_equation.jsonl -TotalCount 1 | python -m json.tool (PS)

'''
{
    "prompt": {
        "default": "You are an expert chemist. Given a chemical equation, please balance the equation without any explanation and maintain the order of reactants and products as given."
    },
    "question": "Here is a unbalanced chemical equation:\nNa2CrO4(aq) + LiBr(aq) = Li2CrO4(aq) + NaBr(s)\nThe balanced chemical equation is:",
    "answer": "Na2CrO4(aq) + 2LiBr(aq) = Li2CrO4(aq) + 2NaBr(s)",
    "type": "filling",
    "domain": "Chemistry",
    "details": {
        "level": "L3",
        "task": "balancing_chemical_equation",
        "subtask": "balancing_chemical_equation",
        "source": "web"
    },
    "answerKey": "",
    "choices": {
        "text": [],
        "label": []
    }
}
'''

Get-Content src/data/tool/eval_real.json -Raw | Select-Object -First 1 | python -m json.tool
'''
....
'''

Get-Content src/data/math/orz_math_13k_collection_hard.json -Raw | Select-Object -First 1 | python -m json.tool
'''
 [
        {
            "from": "human",
            "value": "M\u00e9nehould is organizing a birthday party, which will have 15 girls and 15 boys (including herself). She is preparing a personal label for each guest. In how many ways can she arrange the labels on the seats of the round table while maintaining the alternation of girl-boy?"  
        },
        {
            "from": "assistant",
            "ground_truth": {
                "value": "2\\times(15!)^2"
            }
        }
    ],
'''