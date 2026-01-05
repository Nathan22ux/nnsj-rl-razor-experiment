from datasets import Dataset
import json
from pathlib import Path

from logger import get_logger

logger = get_logger(__name__)

# Get the directory where this file is located
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR


def load_dataset_byname(dataset_name):
    """
    Load datasets from local files.

    Datasets:
    - Math Reasoning: Open-Reasoner-Zero (local JSON)
    - Science Q&A: SciKnowEval Chemistry L3 (local JSONL files)
    - Tool Use: ToolAlpaca (local JSON)
    """
    logger.info("=" * 70)
    logger.info("LOADING DATASETS FROM LOCAL FILES")
    logger.info("=" * 70)

    if dataset_name == 'math':
        # Math Reasoning: Open-Reasoner-Zero (local file)
        logger.info("[1/3] Loading Math Reasoning dataset from local file...")
        math_file = DATA_DIR / "math" / "orz_math_13k_collection_hard.json"

        if not math_file.exists():
            raise FileNotFoundError(f"Math dataset not found at {math_file}")

        logger.info(f"Loading from: {math_file}")
        with open(math_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to list of dicts with '0' and '1' keys (matching original format)
        examples = []
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                examples.append({
                    '0': item[0],
                    '1': item[1]
                })

        dataset = Dataset.from_list(examples)
        logger.info(f"Successfully loaded Open-Reasoner-Zero: {len(dataset)} examples")

    elif dataset_name == 'science':
        # Science Q&A: SciKnowEval Chemistry L3 (local JSONL files)
        logger.info("[2/3] Loading Science Q&A dataset from local files...")
        science_dir = DATA_DIR / "science"

        if not science_dir.exists():
            raise FileNotFoundError(f"Science dataset directory not found at {science_dir}")

        # Load all JSONL files from the Chemistry L3 directory
        all_examples = []
        jsonl_files = list(science_dir.glob("*.jsonl"))

        logger.info(f"Found {len(jsonl_files)} JSONL files in {science_dir}")

        for jsonl_file in jsonl_files:
            logger.info(f"Loading: {jsonl_file.name}")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_examples.append(json.loads(line))

        dataset = Dataset.from_list(all_examples)
        logger.info(f"Successfully loaded SciKnowEval: {len(dataset)} examples")

    elif dataset_name == 'tool':
        # Tool Use: ToolAlpaca (local JSON file)
        logger.info("[3/3] Loading Tool Use dataset from local file...")
        tool_file = DATA_DIR / "tool" / "train_data.json"

        if not tool_file.exists():
            raise FileNotFoundError(f"Tool dataset not found at {tool_file}")

        logger.info(f"Loading from: {tool_file}")
        with open(tool_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded ToolAlpaca: {len(data)} API entries")

        # Flatten the nested structure into training examples
        examples = []

        for row in data:
            api_name = row.get('Name', '')
            api_desc = row.get('Description', '')
            nl_doc = row.get('NLDocumentation', '')

            # Each API has multiple instances
            instances = row.get('Instances', [])

            for instance in instances:
                if not isinstance(instance, dict):
                    continue

                user_input = instance.get('input', '')
                output = instance.get('output', '')
                intermediate_steps = instance.get('intermediate_steps', [])

                if not user_input or not output:
                    continue

                # Build the instruction with API context
                instruction = f"You have access to the following API:\n{api_name}: {api_desc}\n\n"
                if nl_doc:
                    instruction += f"API Documentation:\n{nl_doc}\n\n"
                instruction += "User Request:"

                # Build the full response including tool use steps
                full_response = ""
                for step in intermediate_steps:
                    if isinstance(step, list) and len(step) >= 2:
                        action_info = step[0]
                        observation = step[1] if len(step) > 1 else ""

                        if isinstance(action_info, list) and len(action_info) >= 3:
                            thought = action_info[2]
                            action = action_info[0]
                            action_input = action_info[1]

                            full_response += f"Thought: {thought}\n"
                            full_response += f"Action: {action}\n"
                            full_response += f"Action Input: {action_input}\n"
                            full_response += f"Observation: {observation}\n\n"

                full_response += f"Final Answer: {output}"

                examples.append({
                    'instruction': instruction,
                    'input': user_input,
                    'output': full_response,
                })

        logger.info(f"Extracted {len(examples)} tool-use training examples")

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(examples)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Choose from 'math', 'science', or 'tool'.")

    logger.info("=" * 70)
    logger.info("DATASET LOADING COMPLETE")
    logger.info("=" * 70)

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