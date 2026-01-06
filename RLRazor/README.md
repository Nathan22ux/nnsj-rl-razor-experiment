C:.
|   main.py
|   README.md
|   requirements.txt
|   run_circuit_analysis.py
|   
\---src
    |   logger.py
    |   
    +---circuits
    |   |   checkpoint_loader.py
    |   |   discovery.py
    |   |   test_circuits.py
    |   |   visualization.py
    |   |   
    |   \---__pycache__
    |           discovery.cpython-314.pyc
    |
    +---config
    |       CONFIG.py
    |       
    +---data
    |   |   dataset_utils.py
    |   |   load_data.py
    |   |   
    |   +---math
    |   |   |   orz_math_13k_collection_hard.json
    |   |   |   orz_math_57k_collected.json
    |   |   |   orz_math_72k_collection_extended.json
    |   |   |
    |   |   \---eval_data
    |   |           aime2024.json
    |   |           gpqa_diamond.json
    |   |           math500.json
    |   |
    |   +---science
    |   |       balancing_chemical_equation.jsonl
    |   |       molar_weight_calculation.jsonl
    |   |       molecular_property_calculation.jsonl
    |   |       molecule_structure_prediction.jsonl
    |   |       reaction_prediction.jsonl
    |   |       retrosynthesis.jsonl
    |   |
    |   \---tool
    |           eval_real.json
    |           eval_simulated.json
    |           public_apis.json
    |           train_data.json
    |
    +---evaluation
    |       evaluation.py
    |
    +---models
    |       load_model.py
    |       regularization.py
    |
    +---training
    |       experiment.py
    |       training.py
    |
    +---verification
    |       verify_activation.py
    |       verify_setup.py
    |
    \---visualization
            visualization.py
            visualize_circuits.py