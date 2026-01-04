from datasets import load_dataset, Dataset
import pandas as pd

from logger import get_logger

logger = get_logger(__name__)


def load_dataset_byname(dataset_name):
    """
    Load all datasets for the experiment.

    Datasets:
    - Math Reasoning: Open-Reasoner-Zero
    - Science Q&A: SciKnowEval Chemistry
    - Tool Use: ToolAlpaca
    """
    logger.info("=" * 70)
    logger.info("LOADING DATASETS")
    logger.info("=" * 70)

    if dataset_name == 'math':
        # Math Reasoning: Open-Reasoner-Zero
        logger.info("[1/3] Loading Math Reasoning dataset...")
        try:
            logger.info("Attempting to load Open-Reasoner-Zero from HuggingFace...")
            dataset = load_dataset("Tonic/OpenReasonerZero", split="train")
            logger.info(f"Successfully loaded Open-Reasoner-Zero: {len(dataset)} examples")
            logger.debug(f"Dataset columns: {dataset.column_names if hasattr(dataset, 'column_names') else 'N/A'}")
            logger.debug(f"First example keys: {list(dataset[0].keys()) if len(dataset) > 0 else 'Empty dataset'}")
        except Exception as e:
            logger.warning(f"Open-Reasoner-Zero not available: {str(e)}")
            logger.info("Falling back to GSM8K...")
            dataset = load_dataset("gsm8k", "main", split="train")
            logger.info(f"Successfully loaded GSM8K: {len(dataset)} examples")

    elif dataset_name == 'science':
        # Science Q&A: SciKnowEval Chemistry L-3
        logger.info("[2/3] Loading Science Q&A dataset...")
        try:
            logger.info("Attempting to load SciKnowEval from HuggingFace...")
            dataset = load_dataset("hicai-zju/SciKnowEval", split="train")
            logger.info(f"Successfully loaded SciKnowEval: {len(dataset)} examples")
        except Exception as e:
            logger.warning(f"SciKnowEval not available: {str(e)}")
            logger.info("Falling back to SciQ...")
            dataset = load_dataset("sciq", split="train")
            logger.info(f"Successfully loaded SciQ: {len(dataset)} examples")

    elif dataset_name == 'tool':
        # Tool Use: ToolAlpaca
        logger.info("[3/3] Loading Tool Use dataset...")
        try:
            logger.info("Attempting to load ToolAlpaca from GitHub...")
            tool_url = "https://github.com/tangqiaoyu/ToolAlpaca/raw/main/data/train_data.json"
            df = pd.read_json(tool_url)
            logger.info(f"Downloaded ToolAlpaca: {len(df)} API entries")

            # Flatten the nested structure into training examples
            examples = []

            for _, row in df.iterrows():
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
            Dataset.from_list(examples)

        except Exception as e:
            logger.warning(f"ToolAlpaca not available: {str(e)}")
            dataset = None

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