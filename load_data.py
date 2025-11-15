from datasets import load_dataset, Dataset
import pandas as pd


def load_datasets():
    """
    Load all datasets for the experiment.
    
    Datasets:
    - Math Reasoning: Open-Reasoner-Zero
    - Science Q&A: SciKnowEval Chemistry
    - Tool Use: ToolAlpaca
    """
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    datasets = {}
    
    # Math Reasoning: Open-Reasoner-Zero
    print("\n[1/3] Loading Math Reasoning dataset...")
    try:
        print(" Attempting to load Open-Reasoner-Zero from HuggingFace...")
        math_dataset = load_dataset("Tonic/OpenReasonerZero", split="train")
        print(f" Successfully loaded Open-Reasoner-Zero: {len(math_dataset)} examples")
        # Check dataset values
        print(" Dataset columns:", math_dataset.column_names if hasattr(math_dataset, 'column_names') else 'N/A')
        print(" First example keys:", list(math_dataset[0].keys()) if len(math_dataset) > 0 else 'Empty dataset')
        datasets['math'] = math_dataset
    except Exception as e:
        print(f" Open-Reasoner-Zero not available: {str(e)}")
        print(" Falling back to GSM8K...")
        math_dataset = load_dataset("gsm8k", "main", split="train")
        print(f" Successfully loaded GSM8K: {len(math_dataset)} examples")
        datasets['math'] = math_dataset
    
    # Science Q&A: SciKnowEval Chemistry L-3
    print("\n[2/3] Loading Science Q&A dataset...")
    try:
        print(" Attempting to load SciKnowEval from HuggingFace...")
        science_dataset = load_dataset("Sujal0077/sciknoweval", split="train")
        print(f" Successfully loaded SciKnowEval: {len(science_dataset)} examples")
        datasets['science'] = science_dataset
    except Exception as e:
        print(f" SciKnowEval not available: {str(e)}")
        print(" Falling back to SciQ...")
        science_dataset = load_dataset("sciq", split="train")
        print(f" Successfully loaded SciQ: {len(science_dataset)} examples")
        datasets['science'] = science_dataset
    
    # Tool Use: ToolAlpaca
    print("\n[3/3] Loading Tool Use dataset...")
    try:
        print(" Attempting to load ToolAlpaca from GitHub...")
        tool_url = "https://github.com/tangqiaoyu/ToolAlpaca/raw/main/data/train_data.json"
        tool_dataset = pd.read_json(tool_url)
        print(f" Successfully loaded ToolAlpaca: {len(tool_dataset)} examples")
        datasets['tool'] = tool_dataset
    except Exception as e:
        print(f" ToolAlpaca not available: {str(e)}")
        datasets['tool'] = None
    
    print("\n" + "="*70)
    print("DATASET LOADING COMPLETE")
    print("="*70)
    
    return datasets


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