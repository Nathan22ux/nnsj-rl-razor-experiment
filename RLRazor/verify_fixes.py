"""Quick verification of reward.py boxed extraction and import chain."""
import sys
sys.path.insert(0, 'src')

from trainingv1.reward import extract_answer, check_answer_correctness

print("=" * 50)
print("REWARD VERIFICATION TESTS")
print("=" * 50)

# Test 1: boxed extraction
result = extract_answer(r"blah blah \boxed{42} done")
assert result == "42", f"FAIL boxed basic: got {result!r}"
print("PASS: boxed{42} -> '42'")

# Test 2: boxed with negative decimal
result = extract_answer(r"The answer is \boxed{-3.14}")
assert result == "-3.14", f"FAIL boxed negative: got {result!r}"
print("PASS: boxed{-3.14} -> '-3.14'")

# Test 3: math reward with boxed
assert check_answer_correctness(r"So \boxed{42}", "42", domain="math")
print("PASS: reward boxed math correct")

# Test 4: math reward with text
assert check_answer_correctness("Answer: 42", "42", domain="math")
print("PASS: reward text math correct")

# Test 5: false positive rejection
assert not check_answer_correctness("Answer: 43", "42", domain="math")
print("PASS: false positive rejected")

# Test 6: science reward
assert check_answer_correctness("Answer: C", "C", domain="science")
print("PASS: science MCQ correct")

# Test 7: rollout import chain
from trainingv1.rollout import generate_group_samples, recompute_logprobs
print("PASS: rollout imports OK (generate_group_samples + recompute_logprobs)")

# Test 8: dr_loss import
from trainingv1.dr_loss import dr_grpo_loss
print("PASS: dr_loss imports OK")

# Test 9: advantages import
from trainingv1.advantages import compute_group_advantages
print("PASS: advantages imports OK")

print("=" * 50)
print("ALL VERIFICATION TESTS PASSED")
print("=" * 50)
