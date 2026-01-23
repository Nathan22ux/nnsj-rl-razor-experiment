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

    if pred == gt:
        return True
    
    try:
        p = float(pred)
        g = float(gt)
        return math.isclose(p, g, rel_tol=1e-5, abs_tol=1e-5)
    except:
        return False
    
def correctness_science(pred, gt):
    """
    Science tasks involve molecular properties / reactions.
    Paper uses exact match due to discrete outputs.
    """
    return pred == gt

def correctness_tool(pred, gt):
    """
    Tool tasks (API calls / strings) are exact match.
    """
    return pred == gt

def check_answer_correctness(pred_text, gt, domain="math", use_substring=True):
    """
    pred_text: raw string from rollout
    gt: ground truth from dataset (string)
    domain: {"math","science","tool"}
    use_substring: if True, check if gt appears in pred_text (simpler, more robust)

    Uses hybrid approach:
    1. First tries substring matching (removes whitespace, checks if gt in pred_text)
    2. Falls back to extraction + comparison if substring doesn't match
    """

    # First try substring matching (simpler and more robust)
    if use_substring:
        # Remove whitespace for comparison to handle formatting differences
        pred_clean = pred_text.replace(" ", "").replace("\n", "")
        gt_clean = gt.replace(" ", "").replace("\n", "")

        if gt_clean in pred_clean:
            return True

    # Fallback to extraction-based matching
    pred = extract_answer(pred_text)
    if pred is None or pred == "":
        return False

    if domain == "math":
        return correctness_math(pred, gt)

    if domain == "science":
        return correctness_science(pred, gt)

    if domain == "tool":
        return correctness_tool(pred, gt)

    # default strict
    return pred == gt

def build_binary_rewards(generations, answers, domains=None):
    """
    generations: list[list[str]] -> [prompt][sample]
    answers:     list[str]       -> [prompt]
    domains:     list[str] or None (infer "math" default)

    returns:
        rewards: list[Tensor(group)] with 0/1 values
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