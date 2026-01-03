"""
Unified Dataset Interface for RL's Razor

Handles multiple dataset formats:
- Open-Reasoner-Zero (nested structure with '0', '1' keys)
- GSM8K (question, answer)
- Alpaca (instruction, input, output)
- Natural Questions (for mechanistic work)
- Generic text datasets

Provides consistent interface regardless of source.
"""

from datasets import load_dataset, Dataset
import pandas as pd
from typing import Dict, List, Union, Optional


class UnifiedDatasetInterface:
    """
    Normalize different dataset formats to a standard interface.
    
    Standard format:
    {
        'question': str,  # The question/instruction
        'answer': str,    # The expected answer/output
        'text': str,      # Formatted text for training
    }
    """
    
    @staticmethod
    def detect_format(example: Dict) -> str:
        """
        Detect dataset format from example keys.
        
        Args:
            example: Single example from dataset
        
        Returns:
            str: Format name ('open-reasoner', 'gsm8k', 'alpaca', etc.)
        """
        keys = set(example.keys())
        
        if '0' in keys and '1' in keys:
            return 'open-reasoner'
        elif 'question' in keys and 'answer' in keys:
            return 'gsm8k'
        elif 'instruction' in keys and 'output' in keys:
            return 'alpaca'
        elif 'prompt' in keys and 'completion' in keys:
            return 'completion'
        elif 'text' in keys:
            return 'text'
        else:
            raise ValueError(f"Unknown dataset format. Keys: {keys}")
    
    @staticmethod
    def from_open_reasoner(example: Dict) -> Dict:
        """Convert Open-Reasoner-Zero format"""
        question = example['0']['value']
        
        try:
            answer = example['1']['ground_truth']['value']
        except (KeyError, TypeError):
            answer = str(example['1'])
        
        text = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer: {answer}"
        
        return {
            'question': question,
            'answer': answer,
            'text': text
        }
    
    @staticmethod
    def from_gsm8k(example: Dict) -> Dict:
        """Convert GSM8K format"""
        question = example['question']
        answer = example['answer']
        
        text = f"Question: {question}\nAnswer: {answer}"
        
        return {
            'question': question,
            'answer': answer,
            'text': text
        }
    
    @staticmethod
    def from_alpaca(example: Dict) -> Dict:
        """Convert Alpaca format"""
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']
        
        if input_text:
            question = f"{instruction}\n{input_text}"
            text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
        else:
            question = instruction
            text = f"Instruction: {instruction}\nResponse: {output}"
        
        return {
            'question': question,
            'answer': output,
            'text': text
        }
    
    @staticmethod
    def from_natural_questions(example: Dict) -> Dict:
        """Convert Natural Questions format"""
        question = example['question']['text']
        
        # Extract short answer (NQ has complex structure)
        try:
            # Try short answers first
            if example['annotations']['short_answers']:
                answer_data = example['annotations']['short_answers'][0]
                # Get text from answer_data
                if isinstance(answer_data, dict) and 'text' in answer_data:
                    answer = answer_data['text']
                else:
                    answer = str(answer_data)
            # Fall back to yes/no answer
            elif 'yes_no_answer' in example['annotations']:
                answer = example['annotations']['yes_no_answer']
            else:
                answer = ""
        except (KeyError, TypeError, IndexError):
            answer = ""
        
        text = f"Question: {question}\nAnswer: {answer}"
        
        return {
            'question': question,
            'answer': answer,
            'text': text
        }
    
    @staticmethod
    def from_completion(example: Dict) -> Dict:
        """Convert prompt-completion format"""
        question = example['prompt']
        answer = example['completion']
        text = f"{question}{answer}"
        
        return {
            'question': question,
            'answer': answer,
            'text': text
        }
    
    @staticmethod
    def from_text(example: Dict) -> Dict:
        """Handle text-only format"""
        text = example['text']
        
        return {
            'question': text,
            'answer': '',
            'text': text
        }
    
    @staticmethod
    def normalize_example(example: Dict, format_hint: Optional[str] = None) -> Dict:
        """
        Normalize a single example to standard format.
        
        Args:
            example: Example to normalize
            format_hint: Optional format name (auto-detected if None)
        
        Returns:
            Dict with keys: question, answer, text
        """
        if format_hint is None:
            format_hint = UnifiedDatasetInterface.detect_format(example)
        
        converters = {
            'open-reasoner': UnifiedDatasetInterface.from_open_reasoner,
            'gsm8k': UnifiedDatasetInterface.from_gsm8k,
            'alpaca': UnifiedDatasetInterface.from_alpaca,
            'natural_questions': UnifiedDatasetInterface.from_natural_questions,
            'completion': UnifiedDatasetInterface.from_completion,
            'text': UnifiedDatasetInterface.from_text,
        }
        
        if format_hint not in converters:
            raise ValueError(f"Unknown format: {format_hint}")
        
        return converters[format_hint](example)
    
    @staticmethod
    def normalize_dataset(dataset: Dataset, format_hint: Optional[str] = None) -> Dataset:
        """
        Normalize entire dataset to standard format.
        
        Args:
            dataset: HuggingFace Dataset
            format_hint: Optional format name
        
        Returns:
            Normalized Dataset with standard keys
        """
        # Auto-detect format from first example
        if format_hint is None:
            format_hint = UnifiedDatasetInterface.detect_format(dataset[0])
        
        print(f" Detected dataset format: {format_hint}")
        
        def normalize_batch(examples):
            """Normalize a batch of examples"""
            questions = []
            answers = []
            texts = []
            
            # Process each example in batch
            batch_size = len(next(iter(examples.values())))
            
            for i in range(batch_size):
                # Extract single example from batch
                example = {key: values[i] for key, values in examples.items()}
                
                # Normalize
                normalized = UnifiedDatasetInterface.normalize_example(example, format_hint)
                
                questions.append(normalized['question'])
                answers.append(normalized['answer'])
                texts.append(normalized['text'])
            
            return {
                'question': questions,
                'answer': answers,
                'text': texts
            }
        
        # Apply normalization
        normalized = dataset.map(
            normalize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Normalizing dataset"
        )
        
        print(f" Dataset normalized: {len(normalized)} examples")
        
        return normalized


def load_and_normalize_dataset(dataset_name: str, dataset_config: Optional[str] = None, 
                                split: str = "train", format_hint: Optional[str] = None) -> Dataset:
    """
    Load a dataset from HuggingFace and normalize it.
    
    Args:
        dataset_name: Dataset name on HuggingFace
        dataset_config: Optional dataset configuration
        split: Dataset split to load
        format_hint: Optional format hint
    
    Returns:
        Normalized Dataset
    """
    print(f"\n{'='*70}")
    print(f"LOADING DATASET: {dataset_name}")
    print(f"{'='*70}")
    
    # Load dataset
    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
            print(f" Loaded {dataset_name} ({dataset_config})")
        else:
            dataset = load_dataset(dataset_name, split=split)
            print(f" Loaded {dataset_name}")
        
        print(f"   Size: {len(dataset)} examples")
        print(f"   Columns: {dataset.column_names}")
    
    except Exception as e:
        print(f"✗ Failed to load {dataset_name}: {e}")
        raise
    
    # Normalize
    normalized = UnifiedDatasetInterface.normalize_dataset(dataset, format_hint)
    
    print(f"{'='*70}\n")
    
    return normalized


def load_open_reasoner_zero():
    """Load Open-Reasoner-Zero dataset with fallback to GSM8K"""
    print("Attempting to load Open-Reasoner-Zero...")
    
    try:
        dataset = load_dataset("Tonic/OpenReasonerZero", split="train")
        print(f" Loaded Open-Reasoner-Zero: {len(dataset)} examples")
        return UnifiedDatasetInterface.normalize_dataset(dataset, format_hint='open-reasoner')
    
    except Exception as e:
        print(f" Open-Reasoner-Zero not available: {e}")
        print(" Falling back to GSM8K...")
        
        dataset = load_dataset("gsm8k", "main", split="train")
        print(f" Loaded GSM8K: {len(dataset)} examples")
        return UnifiedDatasetInterface.normalize_dataset(dataset, format_hint='gsm8k')


def load_science_dataset():
    """Load SciKnowEval with fallback to SciQ"""
    print("Attempting to load SciKnowEval...")
    
    try:
        dataset = load_dataset("Sujal0077/sciknoweval", split="train")
        print(f" Loaded SciKnowEval: {len(dataset)} examples")
        # Note: May need custom format handler for SciKnowEval
        return dataset
    
    except Exception as e:
        print(f" SciKnowEval not available: {e}")
        print(" Falling back to SciQ...")
        
        dataset = load_dataset("sciq", split="train")
        print(f" Loaded SciQ: {len(dataset)} examples")
        # SciQ format: question, correct_answer, ...
        
        def format_sciq(example):
            return {
                'question': example['question'],
                'answer': example['correct_answer'],
                'text': f"Question: {example['question']}\nAnswer: {example['correct_answer']}"
            }
        
        return dataset.map(format_sciq, remove_columns=dataset.column_names)


def load_alpaca():
    """Load Alpaca dataset"""
    print("Loading Alpaca dataset...")
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    print(f" Loaded Alpaca: {len(dataset)} examples")
    
    return UnifiedDatasetInterface.normalize_dataset(dataset, format_hint='alpaca')


def load_natural_questions(subset_size: int = 3000):
    """
    Load Natural Questions dataset.
    
    Args:
        subset_size: Number of examples to sample
    
    Returns:
        Normalized dataset
    """
    print(f"Loading Natural Questions (subset: {subset_size})...")
    
    dataset = load_dataset("natural_questions", split="validation")
    print(f" Loaded Natural Questions: {len(dataset)} examples")
    
    # Sample subset
    dataset = dataset.shuffle(seed=42).select(range(min(subset_size, len(dataset))))
    
    return UnifiedDatasetInterface.normalize_dataset(dataset, format_hint='natural_questions')


# Example usage
if __name__ == "__main__":
    print("Testing UnifiedDatasetInterface...\n")
    
    # Test with GSM8K
    print("Test 1: GSM8K")
    gsm8k = load_dataset("gsm8k", "main", split="train[:10]")
    normalized = UnifiedDatasetInterface.normalize_dataset(gsm8k)
    print(f"Sample: {normalized[0]}\n")
    
    # Test with Alpaca
    print("Test 2: Alpaca")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train[:10]")
    normalized = UnifiedDatasetInterface.normalize_dataset(alpaca)
    print(f"Sample: {normalized[0]}\n")
    
    print(" All tests passed!")