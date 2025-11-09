import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import os

# Allow code evaluation for metrics
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def check_gpu():
    """Check GPU availability."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device

MODEL_NAME = "openai-community/gpt2"  # Changed to a smaller model, target LLAMA-3B

def load_model_and_tokenizer(model_name=MODEL_NAME):
    """Load model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {model_name}")
    print(f"Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")
    
    return model, tokenizer

def load_dataset_by_name(dataset_name):
    """Load a specific dataset by name."""
    if dataset_name == "math":
        # Math Reasoning: Open-Reasoner-Zero
        try:
            math_dataset = load_dataset("Tonic/OpenReasonerZero", split="train")
            print(f"Loaded Open-Reasoner-Zero: {len(math_dataset)} examples")
            print("Dataset columns:", math_dataset.column_names if hasattr(math_dataset, 'column_names') else 'N/A')
            print("First example:", math_dataset[0] if len(math_dataset) > 0 else 'Empty dataset')
            return math_dataset
        except:
            print("Warning: Open-Reasoner-Zero not available, using GSM8K")
            return load_dataset("gsm8k", "main", split="train")
    
    elif dataset_name == "science":
        # Science Q&A: SciKnowEval Chemistry L-3
        try:
            science_dataset = load_dataset("Sujal0077/sciknoweval", split="train")
            print(f"Loaded SciKnowEval: {len(science_dataset)} examples")
            return science_dataset
        except:
            print("Warning: SciKnowEval not available, using SciQ")
            return load_dataset("sciq", split="train")
    
    elif dataset_name == "tool":
        # Tool Use: ToolAlpaca
        try:
            tool_url = "https://github.com/tangqiaoyu/ToolAlpaca/raw/main/data/train_data.json"
            tool_dataset = pd.read_json(tool_url)
            print(f"Loaded ToolAlpaca: {len(tool_dataset)} examples")
            return tool_dataset
        except:
            print("Warning: ToolAlpaca not available")
            return None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def format_dataset_for_training(dataset, tokenizer):
    """Format dataset for training."""
    def format_prompt(example):
        # Format depends on dataset structure
        if 'question' in example and 'answer' in example:
            return {"text": f"Question: {example['question']}\n\nAnswer: {example['answer']}"}
        elif 'problem' in example and 'solution' in example:
            return {"text": f"Problem: {example['problem']}\n\nSolution: {example['solution']}"}
        else:
            return {"text": str(example)}
    
    # Check if dataset is a pandas DataFrame
    if isinstance(dataset, pd.DataFrame):
        dataset = Dataset.from_pandas(dataset)
    
    # Format the dataset
    formatted_dataset = dataset.map(format_prompt)
    return formatted_dataset