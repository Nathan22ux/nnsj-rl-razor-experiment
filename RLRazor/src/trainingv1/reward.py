import re
import json
import math

def extract_answer(text):
    """
    Extracts the answer from the given text. Handles various answer formats:
    - "Answer: <answer>"
    - "The answer is <answer>"
    - "So the final answer is <answer>"
    - etc.

    Args:
        text (str): The input text containing the answer.

    Returns:
        str: The extracted answer text, or an empty string if no answer is found.
    """

    # strip leading/trailing whitespace
    text = text.strip()

    # Try to match common answer patterns (case-insensitive)
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s+(.+)',       # "answer is X", "the final answer is X"
        r'(?:the\s+)?answer:\s*(.+)',                        # "Answer: X", "the answer: X"
        r'(?:the\s+)?result\s+is:\s*(.+)',                   # "the result is: X", "result is: X"
        r'(?:therefore|thus|so),?\s+(?:the\s+)?result\s+is:\s*(.+)',  # "Therefore, the result is: X"
        r'(?:therefore|thus|so),?\s+.*?is:\s*(.+)',          # "Therefore, ... is: X"
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            break

    # if there are multiple lines, take the first line
    if "\n" in text:
        text = text.split("\n")[0].strip()

    # Strip trailing punctuation (commas, periods)
    text = text.strip().rstrip(",.").strip()

    return text

# Domain specific reward functions

def correctness_math(pred, gt):
    """
    Computes correctness reward for math problems based on numerical answers.

    Args:
        pred (str): The predicted answer.
        gt (str): The ground truth answer.
    """

    # 1. Substring matching (handles LaTeX, symbolic math)
    pred_clean = pred.replace(" ", "").replace("\n", "")
    gt_clean = gt.replace(" ", "").replace("\n", "")

    if gt_clean in pred_clean:
        return True

    # 2. Extract and compare
    extracted = extract_answer(pred)
    if not extracted:
        return False

    # Exact string match
    if extracted == gt:
        return True

    # Numerical comparison with tolerance
    try:
        pred_num = float(extracted)
        gt_num = float(gt)
        return math.isclose(pred_num, gt_num, rel_tol=1e-5, abs_tol=1e-5)
    except:
        return False


def correctness_science(pred, gt):
    """
    Check answer correctness for SCIENCE domain (chemistry multiple-choice).

    Handles:
    - Ground truth formats: "C" or "C. -2.38"
    - Model outputs: "C", "Answer: C", "C. -2.38", etc.

    The model can output either just the letter or the full answer.

    Args:
        pred: Model's output text
        gt: Ground truth answer (e.g., "C. -2.38" or "C")

    Returns:
        bool: True if correct
    """
    # Extract answer key from ground truth (e.g., "C. -2.38" -> "C")
    gt_key = gt.split(".")[0].strip().upper() if "." in gt else gt.strip().upper()

    # 1. Exact match (with whitespace removed)
    pred_clean = pred.replace(" ", "").replace("\n", "").upper()
    gt_clean = gt.replace(" ", "").replace("\n", "").upper()

    if gt_clean in pred_clean or pred_clean in gt_clean:
        return True

    # 2. Check if answer key appears in various formats
    pred_upper = pred.upper()

    patterns = [
        rf'\b{gt_key}\b',                    # Word boundary: " C " or "C."
        rf'answer[:\s]+{gt_key}',            # "answer: C" or "answer C"
        rf'\({gt_key}\)',                    # "(C)"
        rf'^{gt_key}[.\s]',                  # Starts with "C. " or "C "
        rf'{gt_key}\..*',                    # "C. anything"
    ]

    for pattern in patterns:
        if re.search(pattern, pred_upper):
            return True

    # 3. Extract and compare
    extracted = extract_answer(pred).strip().upper()

    # Check if extracted matches answer key
    if extracted == gt_key:
        return True

    # Check if extracted matches full answer
    if extracted == gt.strip().upper():
        return True

    # Check if extracted starts with answer key
    if extracted.startswith(gt_key + "."):
        return True

    return False


def correctness_tool(pred, gt):
    """
    Check answer correctness for TOOL domain (API calls).

    Handles:
    - Action and Action Input format
    - Substring matching for outputs

    Args:
        pred: Model's output text
        gt: Ground truth answer

    Returns:
        bool: True if correct
    """
    # Substring matching (bidirectional)
    pred_clean = pred.replace(" ", "").replace("\n", "")
    gt_clean = gt.replace(" ", "").replace("\n", "")

    if gt_clean in pred_clean or pred_clean in gt_clean:
        return True

    # Extract and exact match
    extracted = extract_answer(pred)
    if extracted and extracted == gt:
        return True

    return False

def check_answer_correctness(pred, gt, domain="math", use_substring=True):
    """
    Main API for checking answer correctness across all domains.

    Dispatches to domain-specific checking functions:
    - math: Numerical + symbolic answers with tolerance
    - science: Multiple-choice chemistry (handles letter or full answer)
    - tool: API call outputs (exact or substring matching)

    Args:
        pred: Model's full output text
        gt: Ground truth answer from dataset
        domain: One of {"math", "science", "tool"}
        use_substring: Legacy parameter (kept for compatibility)

    Returns:
        bool: True if answer is correct

    Examples:
        >>> check_answer_correctness("Answer: 42", "42", domain="math")
        True
        >>> check_answer_correctness("Answer: C", "C. -2.38", domain="science")
        True
        >>> check_answer_correctness("Action: search", "search query", domain="tool")
        False
    """
    if domain == "math":
        return correctness_math(pred, gt)
    elif domain == "science":
        return correctness_science(pred, gt)
    elif domain == "tool":
        return correctness_tool(pred, gt)
    else:
        # Fallback for unknown domains: simple substring matching
        pred_clean = pred.replace(" ", "").replace("\n", "")
        gt_clean = gt.replace(" ", "").replace("\n", "")
        return gt_clean in pred_clean


def build_binary_rewards(generations, answers, domains=None):
    """
    Build binary reward tensors for GRPO training.

    For each prompt and its group of generations, assigns 1.0 if the
    generation is correct, 0.0 otherwise.

    Args:
        generations: list[list[str]] - [prompt_idx][sample_idx] = generation text
        answers: list[str] - [prompt_idx] = ground truth answer
        domains: list[str] or None - [prompt_idx] = domain name (default: "math")

    Returns:
        list[Tensor] - [prompt_idx] = Tensor of shape [group_size] with 0/1 rewards

    Example:
        >>> generations = [["Answer: 4", "Answer: 5", "Answer: 4"]]
        >>> answers = ["4"]
        >>> rewards = build_binary_rewards(generations, answers)
        >>> rewards[0].tolist()
        [1.0, 0.0, 1.0]
    """
    import torch

    rewards = []
    for i, g_samples in enumerate(generations):
        gt = answers[i]
        domain = domains[i] if domains is not None else "math"

        r = [1.0 if check_answer_correctness(sample, gt, domain) else 0.0
             for sample in g_samples]

        rewards.append(torch.tensor(r, dtype=torch.float32))

    return rewards
