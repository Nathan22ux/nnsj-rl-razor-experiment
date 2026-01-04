from datasets import load_dataset, Dataset
import pandas as pd


def load_dataset_by_name(dataset_name):
    """
    Load a specific dataset for the experiment.

    Args:
        dataset_name: Name of the dataset to load ('math', 'science', or 'tool')

    Returns:
        Loaded dataset

    Datasets:
    - math: Open-Reasoner-Zero (fallback: GSM8K)
    - science: SciKnowEval Chemistry (fallback: SciQ)
    - tool: ToolAlpaca
    """
    print("\n" + "="*70)
    print(f"LOADING {dataset_name.upper()} DATASET")
    print("="*70)

    if dataset_name == 'math':
        # Math Reasoning: Open-Reasoner-Zero
        print("\nLoading Math Reasoning dataset...")
        try:
            print(" Attempting to load Open-Reasoner-Zero from HuggingFace...")
            dataset = load_dataset("Tonic/OpenReasonerZero", split="train")
            print(f" Successfully loaded Open-Reasoner-Zero: {len(dataset)} examples")
            # Check dataset values
            print(" Dataset columns:", dataset.column_names if hasattr(dataset, 'column_names') else 'N/A')
            print(" First example keys:", list(dataset[0].keys()) if len(dataset) > 0 else 'Empty dataset')
        except Exception as e:
            print(f" Open-Reasoner-Zero not available: {str(e)}")
            print(" Falling back to GSM8K...")
            dataset = load_dataset("gsm8k", "main", split="train")
            print(f" Successfully loaded GSM8K: {len(dataset)} examples")

    elif dataset_name == 'science':
        # Science Q&A: SciKnowEval Chemistry L-3
        print("\nLoading Science Q&A dataset...")
        try:
            print(" Attempting to load SciKnowEval from HuggingFace...")
            dataset = load_dataset("Sujal0077/sciknoweval", split="train")
            print(f" Successfully loaded SciKnowEval: {len(dataset)} examples")
        except Exception as e:
            print(f" SciKnowEval not available: {str(e)}")
            print(" Falling back to SciQ...")
            dataset = load_dataset("sciq", split="train")
            print(f" Successfully loaded SciQ: {len(dataset)} examples")

    elif dataset_name == 'tool':
        # Tool Use: ToolAlpaca
        print("\nLoading Tool Use dataset...")
        try:
            print(" Attempting to load ToolAlpaca from GitHub...")
            tool_url = "https://github.com/tangqiaoyu/ToolAlpaca/raw/main/data/train_data.json"
            dataset = pd.read_json(tool_url)
            print(f" Successfully loaded ToolAlpaca: {len(dataset)} examples")
        except Exception as e:
            print(f" ToolAlpaca not available: {str(e)}")
            dataset = None

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Choose from 'math', 'science', or 'tool'.")

    print("\n" + "="*70)
    print("DATASET LOADING COMPLETE")
    print("="*70)

    return dataset


def format_dataset_for_sft(examples):
    """
    Format dataset for SFT training by creating a 'text' field.
    
    Args:
        examples: Batch of examples from the dataset
        
    Returns:
        Dictionary with 'text' field
    """
    texts = []
    for i in range(len(examples['0'])):
        question = examples['0'][i]['value']
        
        # Get answer from ground_truth if available
        try:
            answer = examples['1'][i]['ground_truth']['value']
        except (KeyError, TypeError):
            answer = str(examples['1'][i])
        
        # Format as conversation
        text = f"Question: {question}\nAnswer: {answer}"
        texts.append(text)
    
    return {'text': texts}


def format_dataset_for_grpo(examples):
    """
    Format dataset for GRPO training - needs 'prompt' and 'answer' fields.
    
    Args:
        examples: Batch of examples from the dataset
        
    Returns:
        Dictionary with 'prompt' and 'answer' fields
    """
    prompts = []
    answers = []
    for i in range(len(examples['0'])):
        question = examples['0'][i]['value']
        try:
            answer = examples['1'][i]['ground_truth']['value']
        except (KeyError, TypeError):
            answer = str(examples['1'][i])
        
        prompt = f"Question: {question}\nAnswer:"
        prompts.append(prompt)
        answers.append(answer)
    
    return {'prompt': prompts, 'answer': answers}