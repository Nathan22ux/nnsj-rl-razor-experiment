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
    # If both look like chemical equations, use equation matching
    if is_chemical_equation(pred) and is_chemical_equation(gt):
        return chem_equation_equivalent(pred, gt)

    # Extract answer key from ground truth (e.g., "C. -2.38" -> "C")
    gt_key = gt.split(".")[0].strip().upper() if "." in gt else gt.strip().upper()

    # 1. Substring match only for long ground truths (not single letters like "C")
    gt_clean = gt.replace(" ", "").replace("\n", "").upper()
    pred_clean = pred.replace(" ", "").replace("\n", "").upper()

    if len(gt_clean) > 3 and gt_clean in pred_clean:
        return True

    # 2. Bare-letter match for very short responses (model was told to output only the letter)
    pred_stripped = pred.strip().upper()
    if len(pred_stripped) <= 3 and pred_stripped == gt_key:
        return True

    # 3. For single-letter MCQ answers, only match explicit answer-context patterns
    pred_upper = pred.upper()

    explicit_patterns = [
        rf'answer[:\s]+{gt_key}\b',           # "answer: C" or "answer C"
        rf'answer\s+is\s+\(?{gt_key}\)?',     # "answer is C" or "answer is (C)"
        rf'\({gt_key}\)\s*\Z',                  # "(C)" at end of response (not mid-line)
        rf'^\s*{gt_key}[.)]\s',               # Starts with "C. " or "C) "
        rf'option\s+{gt_key}\b',              # "option C"
        rf'choice\s+{gt_key}\b',              # "choice C"
        rf'correct\s+(?:answer\s+is\s+)?{gt_key}\b',  # "correct answer is C" or "correct C"
    ]

    for pattern in explicit_patterns:
        if re.search(pattern, pred_upper, re.MULTILINE):
            return True

    # 4. Extract and compare
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


def _normalize_chem_text_reward(text):
    """
    Normalize chemical equation text for reward comparison:
    - Convert arrow variants (→, ->, ⟶, ⇒) to =
    - Convert Unicode subscripts/superscripts to ASCII digits
    """
    s = str(text)
    sub_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    s = s.translate(sub_map)
    sup_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    s = s.translate(sup_map)
    for arrow in ['⟶', '→', '->', '⇒', '=>']:
        s = s.replace(arrow, '=')
    return s.strip()


def is_chemical_equation(text):
    """Heuristic check for chemical equations (used for science rewards).
    Handles arrow variants (→, ->, ⟶, ⇒) in addition to '='.
    """
    if text is None:
        return False
    s = _normalize_chem_text_reward(str(text))
    if "=" not in s:
        return False
    has_elem = re.search(r"[A-Z][a-z]?", s) is not None
    has_terms = "+" in s or s.count("=") >= 1
    return has_elem and has_terms


def _strip_state_annotations(s):
    return re.sub(r"\((aq|s|l|g)\)", "", s, flags=re.IGNORECASE)


def _normalize_chem_term(term):
    t = term.strip()
    t = _strip_state_annotations(t)
    t = t.replace(" ", "")
    m = re.match(r"^(\d+)(.+)$", t)
    if m:
        coeff = int(m.group(1))
        formula = m.group(2)
    else:
        coeff = 1
        formula = t
    return formula, coeff


def _normalize_chem_side(side):
    parts = [p for p in side.split("+") if p.strip()]
    out = {}
    for p in parts:
        formula, coeff = _normalize_chem_term(p)
        if not formula:
            continue
        out[formula] = out.get(formula, 0) + coeff
    return out


def chem_equation_equivalent(prediction, expected):
    # Normalize arrow variants before splitting
    pred_norm = _normalize_chem_text_reward(prediction)
    exp_norm = _normalize_chem_text_reward(expected)
    if not (is_chemical_equation(pred_norm) and is_chemical_equation(exp_norm)):
        return False
    try:
        pred_lhs, pred_rhs = pred_norm.split("=", 1)
        exp_lhs, exp_rhs = exp_norm.split("=", 1)
    except ValueError:
        return False
    pred_left = _normalize_chem_side(pred_lhs)
    pred_right = _normalize_chem_side(pred_rhs)
    exp_left = _normalize_chem_side(exp_lhs)
    exp_right = _normalize_chem_side(exp_rhs)
    return pred_left == exp_left and pred_right == exp_right


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