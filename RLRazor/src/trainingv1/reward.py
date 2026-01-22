import re
import json
import math

def extract_answer(text):
    """
    Extracts the answer from the given text. The answer is expected to be in the format:
    "Answer: <answer_text>"

    Args:
        text (str): The input text containing the answer.

    Returns:
        str: The extracted answer text, or an empty string if no answer is found.
    """
    
    # strip leading/trailing whitespace
    text = text.strip()

    # split on common seprators
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    if "answer:" in text:
        text = text.split("answer:")[-1].strip()
    
    # if there are multiple token, take the last token
    if "\n" in text:
        text = text.split("\n")[0].strip()
    if ":" in text and len(text.split(":")[-1]) < 40:
        text = text.split(":")[-1].strip()
    
    text = text.strip().strip(",").strip(".") # numeric answers
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

def check_answer_correctness(pred_text, gt, domain="math"):
    """
    pred_text: raw string from rollout
    gt: ground truth from dataset (string)
    domain: {"math","science","tool"}
    """

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