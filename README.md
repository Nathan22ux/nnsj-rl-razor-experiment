# nnsj-rl-razor-experiment

**Running Commands**

*Command: 1*
bash
```
pip install -r requirements.txt
```

*Command: 2*
bash
```
python src/main.py
```

*Command: 3*
bash 
```
python src/run_circuit_analysis.py \
    --task math \
    --sft_checkpoint ./results/sft_lr3e-05_bs32/checkpoint-final \
    --rl_checkpoint ./results/grpo_lr2e-05/checkpoint-final \
    --max_examples 50
```

--task: The domain to analyze (math, science, tool).
--sft_checkpoint: Path to best SFT model.
--rl_checkpoint: Path to best RL model.
--top_k_heads: Number of important heads to analyze

*Command: 4*
bash
```
python src/visualize_circuits.py results/circuits/circuit_analysis_math.json
```

- **Forward KL Divergence (Forgetting Law)**
$$KL(P_{base} || P_{model}) = \sum P_{base}(x) \cdot \log\left(\frac{P_{base}(x)}{P_{model}(x)}\right)$$

- **New Task (NT) & Prior Task (PT) Accuracy**
$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(y_{pred} == y_{true})$$

- **Mechanistic Analysis Formulas ("The Razor")**
$$\text{Score}(h) = \frac{P_{patch} - P_{org}}{P_{org}}$$

- **Cross-Model Activation Patching (CMAP) / $\Delta F$**
$$\Delta F = P_{patched} - P_{base}$$

- **Vulnerability Score**
$$\text{Vulnerability} = \Delta F_{RL} - \Delta F_{SFT}$$

- **Circuit Faithfulness***
$$\text{Faithfulness} = \frac{\text{Performance}_{\text{Circuit Only}}}{\text{Performance}_{\text{Full Model}}}$$

- **Training & Regularization Formulas**
$$L_{total} = L_{SFT} + \lambda \sum_{h \in V} ||a^\pi_h - a^{\pi0}_h||^2$$

