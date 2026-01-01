# Folder Structure

RLRazor/
├── src/
│   ├── config/
│   │   └── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── dataset_utils.py
│   ├── models/
│   │   ├── load_model.py
│   │   └── regularization.py
│   ├── training/
│   │   ├── training.py
│   │   └── experiment.py
│   ├── evaluation/
│   │   └── evaluation.py
│   ├── circuits/
│   │   ├── discovery.py
│   │   ├── checkpoint_loader.py
│   │   ├── visualization.py
│   │   └── test_circuits.py
│   ├── visualization/
│   │   ├── visualization.py
│   │   └── visualize_circuits.py
│   ├── verification/
│   │   ├── verify_setup.py
│   │   └── verify_activation.py
│   ├── main.py
│   └── run_circuit_analysis.py
├── notebooks/
│   └── rls_razor_replication.ipynb
├── results/
│   ├── results_math.json
│   ├── grpo_lr3e-05/
│   └── sft_lr3e-05_bs16/
│       ├── README.md
│       └── checkpoint-7/
│           ├── config.json
│           ├── generation_config.json
│           ├── merges.txt
│           ├── model.safetensors
│           ├── optimizer.pt
│           ├── rng_state.pth
│           ├── scheduler.pt
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           ├── trainer_state.json
│           └── vocab.json
├── tests/
│   └── test_all_fixes.py
├── requirements.txt
└── README.md