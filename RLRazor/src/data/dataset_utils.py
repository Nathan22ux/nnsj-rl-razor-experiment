"""
Unified Dataset Interface for RL's Razor

Handles dataset formats used in this project:
- Open-Reasoner-Zero (nested structure with '0', '1' keys)
- SciKnowEval (question, answer format)
- Alpaca/ToolAlpaca (instruction, input, output)

This module provides centralized dataset formatting used by:
- training.py: SFT and GRPO training
- experiment.py: KL divergence computation
- load_data.py: Initial dataset loading from local files

All datasets are normalized to a standard format with 'question', 'answer', and 'text' fields.
"""

from datasets import Dataset
from typing import Dict, Optional


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
            str: Format name ('open-reasoner', 'sciknoweval', 'alpaca')
        """
        keys = set(example.keys())

        if '0' in keys and '1' in keys:
            return 'open-reasoner'
        elif 'question' in keys and 'answer' in keys:
            return 'sciknoweval'
        elif 'instruction' in keys and 'output' in keys:
            return 'alpaca'
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

        # Prompt format for math reasoning tasks (without answer)
        prompt = f"Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"
        # Training text includes the answer
        text = f"{prompt} {answer}"

        return {
            'question': question,
            'answer': answer,
            'text': text,
            'prompt': prompt  # Store the prompt template for evaluation
        }
    
    @staticmethod
    def from_sciknoweval(example: Dict) -> Dict:
        """Convert SciKnowEval format

        Dataset structure:
        - question: The question text
        - answerKey: The correct answer key (e.g., "A", "B", "C", "D")
        - choices.text: List of answer values (e.g., ["3469.900", "3740.400", ...])
        - choices.label: List of answer labels (e.g., ["A", "B", "C", "D"])
        - answer: Empty string (not used)
        """
        question = example['question']

        # Get the correct answer by mapping answerKey to choices
        answer_key = example.get('answerKey', '')

        if answer_key and 'choices' in example:
            # Find the index of the answer key in choices.label
            labels = example['choices'].get('label', [])
            values = example['choices'].get('text', [])

            try:
                answer_idx = labels.index(answer_key)
                answer = f"{answer_key}. {values[answer_idx]}"  # e.g., "B. 3740.400"
            except (ValueError, IndexError):
                # Fallback if answerKey not found
                answer = answer_key
        else:
            # Fallback to the answer field (though it's usually empty)
            answer = example.get('answer', '')

        # Include the prompt instructions if available
        prompt_instructions = ""
        if 'prompt' in example and isinstance(example['prompt'], dict):
            prompt_instructions = example['prompt'].get('default', '')

        # Build the prompt template (without answer) for evaluation
        if prompt_instructions:
            prompt = f"{prompt_instructions}\n{question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        # Training text includes the answer
        text = f"{prompt} {answer}"

        return {
            'question': question,
            'answer': answer,
            'text': text,
            'prompt': prompt  # Store the prompt template for evaluation
        }

    @staticmethod
    def from_alpaca(example: Dict) -> Dict:
        """Convert Alpaca format"""
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']

        # Build the prompt template (without answer) for evaluation
        if input_text:
            question = f"{instruction}\n{input_text}"
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            question = instruction
            prompt = f"Instruction: {instruction}\nResponse:"

        # Training text includes the answer
        text = f"{prompt} {output}"

        return {
            'question': question,
            'answer': output,
            'text': text,
            'prompt': prompt  # Store the prompt template for evaluation
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
            'sciknoweval': UnifiedDatasetInterface.from_sciknoweval,
            'alpaca': UnifiedDatasetInterface.from_alpaca,
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
            prompts = []

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
                prompts.append(normalized['prompt'])

            return {
                'question': questions,
                'answer': answers,
                'text': texts,
                'prompt': prompts
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



# Example usage
if __name__ == "__main__":
    print("Testing UnifiedDatasetInterface with local datasets...\n")
    print("Note: This module is now actively used by the training pipeline.")
    print("See training.py and experiment.py for usage examples.")
    print("\nTo test, run from project root:")
    print("  python -c \"from data.load_data import load_dataset_byname; from data.dataset_utils import UnifiedDatasetInterface; ds = load_dataset_byname('science'); normalized = UnifiedDatasetInterface.normalize_dataset(ds.select(range(3))); print(normalized[0])\"")