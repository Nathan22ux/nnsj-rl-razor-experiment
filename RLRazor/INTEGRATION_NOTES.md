# Integration Notes: trainingv1 → Experiment Pipeline

This document explains the integration of `trainingv1/` implementations into the main experiment pipeline.

## Files Created

1. **[src/training/experiment_v2.py](src/training/experiment_v2.py)** - New experiment file using trainingv1 implementations
2. **[main_v2.py](main_v2.py)** - New entry point for running experiments with trainingv1

## Files Modified

1. **[src/trainingv1/train_dr_grpo.py](src/trainingv1/train_dr_grpo.py)**
   - Added `domain` parameter for dataset-specific reward checking
   - Changed return value from `(model, π_models)` to `(model, NT)`
   - Updated `check_answer_correctness()` call to pass domain parameter

2. **[src/trainingv1/train_sft_baseline.py](src/trainingv1/train_sft_baseline.py)**
   - No changes needed (already compatible)

## Key Differences: Original vs trainingv1

### Original ([main.py](main.py) + [src/training/](src/training/))
- Uses TRL's `GRPOTrainer` (library implementation)
- May not exactly match Dr.GRPO from paper
- Less control over training details
- Complete infrastructure for sweeps and evaluation

### trainingv1 ([main_v2.py](main_v2.py) + [src/trainingv1/](src/trainingv1/))
- **Custom Dr.GRPO implementation** matching paper exactly
- **Fixed log probability calculation** (only generated tokens, sum not average)
- **Domain-specific reward functions** (math/science/tool)
- **All code review bugs fixed**
- Same infrastructure for sweeps and evaluation

## How to Use

### Run with original implementation:
```bash
python main.py --mode minimal --dataset math
```

### Run with trainingv1 implementation:
```bash
python main_v2.py --mode minimal --dataset math
```

## Integration Details

### 1. Domain Detection
```python
domain_map = {
    "math": "math",
    "science": "science",
    "tool": "tool"
}
domain = domain_map.get(dataset_name, "math")
```

The domain is automatically detected from the dataset name and passed to Dr.GRPO for correct reward checking.

### 2. Function Signatures

**train_sft_baseline:**
```python
def train_sft_baseline(
    model, tokenizer, dataset,
    learning_rate, batch_size, epochs,
    max_samples=3000, eval_dataset=None
) -> (model, NT)
```

**train_dr_grpo:**
```python
def train_dr_grpo(
    model, tokenizer, dataset,
    eval_dataset=None, domain="math",
    μ_iterations=2, lr=2e-5,
    group_size=64, prompts_per_gen=8,
    target_nt=None, max_samples=3000
) -> (model, NT)
```

### 3. Batch Size Handling

**SFT:**
- Uses `gradient_accumulation_steps = 4`
- Effective batch size = `bs * 4`
- Results stored with effective batch size

**RL (Dr.GRPO):**
- No gradient accumulation
- Uses `prompts_per_gen` parameter (set to `bs`)
- Results stored with `bs` directly

### 4. Reward Checking

**Before (broken for science/tool):**
```python
r_group = [1.0 if check_answer_correctness(sample, answer) else 0.0
           for sample in g]
# Always used default domain="math"
```

**After (works for all domains):**
```python
r_group = [1.0 if check_answer_correctness(sample, answer, domain=domain) else 0.0
           for sample in g]
# Uses correct domain-specific checking
```

## Critical Fixes Applied

### 1. Log Probability Calculation
- **Original bug**: Used `-out.loss` (average over all tokens including prompt)
- **Fixed**: Manually compute log probs only on generated tokens, return sum

### 2. Reward Correctness
- **Original bug**: Science answers like "C. -2.38" vs "C" didn't match
- **Fixed**: Domain-specific functions handle all formats

### 3. Array Indexing
- **Original bug**: `batch_answers[i+k]` caused IndexError
- **Fixed**: `batch_answers[k]` (batch is already sliced)

### 4. Return Values
- **Original bug**: `train_dr_grpo` returned list of models
- **Fixed**: Returns `(model, NT)` to match experiment expectations

## Results Files

- Original: `results/results_{dataset}_{mode}.json`
- trainingv1: `results/results_{dataset}_{mode}_v2.json`

Files are separate to allow comparison between implementations.

## Testing

All test scripts verified to work with trainingv1:
- ✓ [test_reward.py](test_reward.py) - Math dataset
- ✓ [test_reward_science.py](test_reward_science.py) - Science dataset
- ✓ [test_reward_tool.py](test_reward_tool.py) - Tool dataset
- ✓ [test_science_training.py](test_science_training.py) - Science SFT training
- ✓ [test_tool_training.py](test_tool_training.py) - Tool SFT training

## Recommendation

**Use main_v2.py with trainingv1 for the actual experiment** because:
1. ✓ Paper-faithful Dr.GRPO implementation
2. ✓ All critical bugs fixed
3. ✓ Correct domain-specific reward checking
4. ✓ Proper log probability calculation
5. ✓ Full test coverage

The original [main.py](main.py) is preserved for reference and comparison.
