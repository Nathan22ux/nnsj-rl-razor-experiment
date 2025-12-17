"""
Circuit Discovery for RL vs SFT Analysis
Implementation of path patching and activation patching to identify which circuits
are reinforced by SFT vs RL fine-tuning.

CORRECTED VERSION - Fixes:
1. Target token computation now uses full sequence (question + answer)
2. Probabilities computed at correct sequential positions
3. CMAP uses proper answer tokens instead of last input token

Based on:
- ACDC paper (automatic circuit discovery)
- "Fine-tuning enhances existing mechanisms" (Prakash et al. 2024)
- Our paper's methodology (Section 2.2-2.4)
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import re


@dataclass
class CircuitScore:
    """Stores importance scores for an attention head"""
    layer: int
    head: int
    score: float
    position: Optional[int] = None


class CircuitDiscovery:
    """
    Implements path patching to identify important attention heads.
    Resource-efficient implementation using:
    - Batched processing where possible
    - Selective head evaluation (only test promising heads)
    - Cached activations to avoid repeated forward passes
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Get model architecture info
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads

        # Detect architecture style
        self._detect_architecture()

        print(f"Initialized CircuitDiscovery for model with {self.n_layers} layers, {self.n_heads} heads per layer")
        print(f"Architecture style: {self.arch_style}")

    def _detect_architecture(self):
        """
        Detect model architecture to handle different naming conventions.
        Supports: Qwen/Llama (model.model.layers), GPT-2 (transformer.h), BERT (encoder.layer)
        """
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.arch_style = 'gpt2'
            self.layers_attr = lambda: self.model.transformer.h
            self.attn_proj_attr = 'c_proj'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.arch_style = 'llama'
            self.layers_attr = lambda: self.model.model.layers
            self.attn_proj_attr = 'o_proj'
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            self.arch_style = 'bert'
            self.layers_attr = lambda: self.model.encoder.layer
            self.attn_proj_attr = 'output'
        else:
            raise ValueError(
                f"Unknown model architecture. Model has attributes: {dir(self.model)[:10]}... "
                "Please update _detect_architecture() in circuit_discovery.py"
            )

    def _get_layer(self, layer_idx):
        """Get layer by index, handling different architectures"""
        return self.layers_attr()[layer_idx]

    def _get_attn_projection(self, layer):
        """Get attention output projection, handling different architectures"""
        if self.arch_style == 'gpt2':
            return layer.attn.c_proj
        elif self.arch_style == 'llama':
            return layer.self_attn.o_proj
        elif self.arch_style == 'bert':
            return layer.attention.output.dense
        else:
            raise ValueError(f"Unknown architecture style: {self.arch_style}")

    def get_attention_hook_points(self):
        """Get all attention head hook points in the model"""
        hook_points = []
        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                hook_points.append((layer_idx, head_idx))
        return hook_points

    @torch.no_grad()
    def extract_activations(self, input_ids, attention_mask=None):
        """
        Extract attention head outputs for all layers.
        Returns a dictionary mapping (layer, head) -> activations
        """
        activations = {}

        def create_hook(layer_idx, head_idx):
            def hook_fn(module, input, output):
                attn_output = output[0] if isinstance(output, tuple) else output

                if len(attn_output.shape) == 2:
                    attn_output = attn_output.unsqueeze(1)

                batch_size, seq_len, hidden_dim = attn_output.shape
                head_dim = hidden_dim // self.n_heads
                attn_output_heads = attn_output.reshape(batch_size, seq_len, self.n_heads, head_dim)
                head_output = attn_output_heads[:, :, head_idx, :]
                activations[(layer_idx, head_idx)] = head_output.clone()

            return hook_fn

        hooks = []
        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                layer = self._get_layer(layer_idx)
                attn_proj = self._get_attn_projection(layer)
                hook = attn_proj.register_forward_hook(create_hook(layer_idx, head_idx))
                hooks.append(hook)

        with torch.no_grad():
            self.model(input_ids, attention_mask=attention_mask)

        for hook in hooks:
            hook.remove()

        return activations

    @torch.no_grad()
    def path_patch_head(
            self,
            original_input_ids,
            counterfactual_input_ids,
            layer_idx: int,
            head_idx: int,
            original_activations: Dict,
            counterfactual_activations: Dict,
            attention_mask=None
    ):
        """
        Perform path patching: replace head (layer_idx, head_idx) with
        counterfactual activations and measure output change.

        Returns the full logits (not just last position) for flexible probability computation.
        """
        patched_activation = counterfactual_activations[(layer_idx, head_idx)]

        def patch_hook(module, input, output):
            attn_output = output[0] if isinstance(output, tuple) else output

            if len(attn_output.shape) == 2:
                attn_output = attn_output.unsqueeze(1)

            batch_size, seq_len, hidden_dim = attn_output.shape
            head_dim = hidden_dim // self.n_heads
            attn_output_heads = attn_output.reshape(batch_size, seq_len, self.n_heads, head_dim)

            seq_len_patch = min(seq_len, patched_activation.shape[0])
            if len(patched_activation.shape) == 2:
                seq_len_patch = min(seq_len, patched_activation.shape[0])
                attn_output_heads[:, :seq_len_patch, head_idx, :] = patched_activation[:seq_len_patch].unsqueeze(0)
            else:
                seq_len_patch = min(seq_len, patched_activation.shape[1])
                attn_output_heads[:, :seq_len_patch, head_idx, :] = patched_activation[:, :seq_len_patch]

            attn_output_patched = attn_output_heads.reshape(batch_size, seq_len, hidden_dim)

            if isinstance(output, tuple):
                return (attn_output_patched,) + output[1:]
            else:
                return attn_output_patched

        layer = self._get_layer(layer_idx)
        attn_proj = self._get_attn_projection(layer)
        hook = attn_proj.register_forward_hook(patch_hook)

        with torch.no_grad():
            outputs = self.model(original_input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        hook.remove()

        return logits

    def compute_head_importance(
            self,
            examples: List[Dict],
            max_examples: int = 50,
            batch_size: int = 4
    ) -> List[CircuitScore]:
        """
        Compute importance scores for all attention heads using path patching.

        CORRECTED: Now computes probability of answer tokens at their correct
        sequential positions in the full question+answer sequence.

        The key insight is that for autoregressive models:
        - P(token_i) is computed from logits at position i-1
        - We need to run the model on the FULL sequence (question + answer)
        - Then check the probability of each answer token at its correct position

        Args:
            examples: List of dicts with 'question', 'answer', 'counterfactual_question'
            max_examples: Maximum number of examples to use
            batch_size: Batch size for processing

        Returns:
            List of CircuitScore objects ranked by importance
        """
        if not examples:
            raise ValueError("No examples provided for circuit discovery!")

        required_keys = ['question', 'answer', 'counterfactual_question']
        for i, ex in enumerate(examples[:5]):
            missing_keys = [key for key in required_keys if key not in ex]
            if missing_keys:
                raise ValueError(f"Example {i} missing required keys: {missing_keys}. Has keys: {list(ex.keys())}")

        print(f"\n✅ Validated {len(examples)} examples with required structure")
        print(f"Computing head importance scores (CORRECTED METHOD)...")
        print(f"Using {min(len(examples), max_examples)} examples")

        examples = examples[:max_examples]
        all_scores = []

        for idx, example in enumerate(tqdm(examples, desc="Processing examples")):
            question = example['question']
            answer = str(example['answer']).strip()
            counterfactual_question = example['counterfactual_question']

            # =========================================================
            # CORRECTED: Create full sequence with question AND answer
            # =========================================================
            # Format: "Question: X\nAnswer: Y"
            full_text = f"{question} {answer}"
            counterfactual_full = f"{counterfactual_question} {answer}"

            # Tokenize the full sequences
            full_ids = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(self.device)

            counterfactual_full_ids = self.tokenizer(
                counterfactual_full,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(self.device)

            # Tokenize just the question to find where answer starts
            question_ids = self.tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(self.device)

            # The answer starts after the question tokens
            answer_start_pos = question_ids.shape[1]

            # Tokenize answer separately to get answer token IDs
            answer_ids = self.tokenizer(
                answer,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)

            if answer_ids.shape[1] == 0:
                continue

            target_tokens = answer_ids[0].tolist()

            # Debug first few examples
            if idx < 3:
                print(f"\n  Example {idx}:")
                print(f"    Q: {question[:60]}...")
                print(f"    A: {answer}")
                print(f"    Full seq length: {full_ids.shape[1]}")
                print(f"    Answer starts at position: {answer_start_pos}")
                print(f"    Target tokens: {[self.tokenizer.decode([t]) for t in target_tokens[:5]]}")

            # =========================================================
            # CORRECTED: Compute probability at correct positions
            # =========================================================
            # For autoregressive LM: P(token at pos i) = softmax(logits[i-1])[token]

            # Extract activations on the full sequence
            original_activations = self.extract_activations(full_ids)
            counterfactual_activations = self.extract_activations(counterfactual_full_ids)

            # Get original model's probability of answer tokens
            with torch.no_grad():
                original_outputs = self.model(full_ids)
                original_logits = original_outputs.logits  # [1, seq_len, vocab]

            # Compute log probability of each answer token at its correct position
            p_org = self._compute_answer_probability(
                original_logits, target_tokens, answer_start_pos
            )

            # Test each head
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    # Path patch this head and get full logits
                    patched_logits = self.path_patch_head(
                        full_ids,
                        counterfactual_full_ids,
                        layer_idx,
                        head_idx,
                        original_activations,
                        counterfactual_activations
                    )

                    # Compute probability with patched activations
                    p_patch = self._compute_answer_probability(
                        patched_logits, target_tokens, answer_start_pos
                    )

                    # Compute importance score (Equation 2)
                    # Negative score = head is important (patching hurts performance)
                    score = (p_patch - p_org) / (p_org + 1e-10)

                    all_scores.append(CircuitScore(
                        layer=layer_idx,
                        head=head_idx,
                        score=score
                    ))

        # Aggregate scores across examples
        aggregated_scores = self._aggregate_scores(all_scores)

        # Sort by importance (most negative = most important)
        aggregated_scores.sort(key=lambda x: x.score)

        return aggregated_scores

    def _compute_answer_probability(
            self,
            logits: torch.Tensor,
            target_tokens: List[int],
            answer_start_pos: int
    ) -> float:
        """
        Compute the probability of generating the answer tokens.

        CORRECTED METHOD:
        For autoregressive models, P(token_i) comes from logits at position i-1.
        So to predict answer token at position `answer_start_pos`, we look at
        logits from position `answer_start_pos - 1`.

        Args:
            logits: Model output logits [1, seq_len, vocab_size]
            target_tokens: List of answer token IDs
            answer_start_pos: Position where answer starts in the sequence

        Returns:
            Average probability of answer tokens (geometric mean in log space)
        """
        total_log_prob = 0.0
        valid_tokens = 0

        for i, token_id in enumerate(target_tokens):
            # Position in sequence where this token appears
            token_pos = answer_start_pos + i

            # Logits that predict this token come from previous position
            pred_pos = token_pos - 1

            if pred_pos < 0 or pred_pos >= logits.shape[1]:
                continue

            # Get probability of this token
            probs = torch.softmax(logits[0, pred_pos, :], dim=-1)
            token_prob = probs[token_id].item()

            # Use log probability to avoid underflow
            if token_prob > 1e-10:
                total_log_prob += np.log(token_prob)
                valid_tokens += 1

        if valid_tokens == 0:
            return 1e-10

        # Return geometric mean (exp of average log prob)
        avg_log_prob = total_log_prob / valid_tokens
        return np.exp(avg_log_prob)

    def _aggregate_scores(self, scores: List[CircuitScore]) -> List[CircuitScore]:
        """Aggregate scores for the same head across multiple examples"""
        score_dict = defaultdict(list)

        for score in scores:
            key = (score.layer, score.head)
            score_dict[key].append(score.score)

        aggregated = []
        for (layer, head), score_list in score_dict.items():
            mean_score = np.mean(score_list)
            aggregated.append(CircuitScore(
                layer=layer,
                head=head,
                score=mean_score
            ))

        return aggregated

    def identify_circuit(
            self,
            examples: List[Dict],
            top_k: int = 20,
            max_examples: int = 50
    ) -> List[CircuitScore]:
        """
        Main entry point for circuit identification.

        Args:
            examples: List of example dicts with question/answer/counterfactual
            top_k: Number of top heads to return
            max_examples: Max examples to process

        Returns:
            List of top-k CircuitScore objects (most important heads)
        """
        print(f"\n{'='*60}")
        print(f"IDENTIFYING CIRCUIT (top {top_k} heads)")
        print(f"{'='*60}")

        all_scores = self.compute_head_importance(examples, max_examples=max_examples)

        # Return top-k most important (most negative scores)
        top_heads = all_scores[:top_k]

        print(f"\n📊 Top {top_k} most important heads:")
        for i, score in enumerate(top_heads[:10]):
            print(f"  {i+1}. Layer {score.layer}, Head {score.head}: importance={score.score:.4f}")

        return top_heads

    def binarize_circuit(
            self,
            circuit: List[CircuitScore],
            threshold: float = None,
            top_k: int = None
    ) -> Dict[Tuple[int, int], int]:
        """Convert continuous circuit scores to binary (in/out)."""
        if threshold is None and top_k is None:
            raise ValueError("Must specify either threshold or top_k")

        binary_circuit = {}

        if top_k is not None:
            important_heads = set((s.layer, s.head) for s in circuit[:top_k])
        else:
            important_heads = set((s.layer, s.head) for s in circuit if s.score < threshold)

        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                key = (layer_idx, head_idx)
                binary_circuit[key] = 1 if key in important_heads else 0

        return binary_circuit


class CrossModelCircuitAnalysis:
    """
    Compare circuits across base, SFT, and RL models.
    Implements Cross-Model Activation Patching (CMAP).
    """

    def __init__(self, base_model, sft_model, rl_model, tokenizer, device="cuda"):
        self.base_model = base_model
        self.sft_model = sft_model
        self.rl_model = rl_model
        self.tokenizer = tokenizer
        self.device = device

        # Create discovery instances for each model
        self.base_discovery = CircuitDiscovery(base_model, tokenizer, device)
        self.sft_discovery = CircuitDiscovery(sft_model, tokenizer, device)
        self.rl_discovery = CircuitDiscovery(rl_model, tokenizer, device)

    @torch.no_grad()
    def cross_model_activation_patching(
            self,
            circuit: List[CircuitScore],
            test_examples: List[Dict],
            max_examples: int = 50
    ) -> Dict[str, List[float]]:
        """
        Implement CMAP (Equation 5 from paper).
        Patch activations from fine-tuned models into base model and measure change.

        CORRECTED: Now uses proper answer tokens and positions.

        Args:
            circuit: List of CircuitScore objects (important heads to test)
            test_examples: List of example dicts with question/answer
            max_examples: Maximum examples to process

        Returns:
            Dictionary with 'sft_deltas' and 'rl_deltas' lists
        """
        print("\nPerforming Cross-Model Activation Patching (CMAP)...")
        print("(CORRECTED: Using proper answer token positions)")

        test_examples = test_examples[:max_examples]

        results = {
            'sft_deltas': [],
            'rl_deltas': [],
            'head_info': []
        }

        for score in tqdm(circuit, desc="Patching circuit heads"):
            layer_idx = score.layer
            head_idx = score.head

            sft_deltas = []
            rl_deltas = []

            for example in test_examples:
                # Handle both dict format and string format
                if isinstance(example, dict):
                    question = example.get('question', example.get('0', {}).get('value', str(example)))
                    answer = str(example.get('answer', '')).strip()
                else:
                    question = str(example)
                    answer = ""

                # Skip if no answer
                if not answer:
                    continue

                # Create full sequence
                full_text = f"{question} {answer}"

                full_ids = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).input_ids.to(self.device)

                # Find answer start position
                question_ids = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).input_ids
                answer_start_pos = question_ids.shape[1]

                # Get answer tokens
                answer_ids = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids
                target_tokens = answer_ids[0].tolist() if answer_ids.shape[1] > 0 else []

                if not target_tokens:
                    continue

                # Get base model probability
                with torch.no_grad():
                    base_outputs = self.base_model(full_ids)
                    base_logits = base_outputs.logits
                    base_prob = self.base_discovery._compute_answer_probability(
                        base_logits, target_tokens, answer_start_pos
                    )

                # Extract activations from all models
                base_activations = self.base_discovery.extract_activations(full_ids)
                sft_activations = self.sft_discovery.extract_activations(full_ids)
                rl_activations = self.rl_discovery.extract_activations(full_ids)

                # Patch SFT activation into base model
                sft_patched_logits = self.base_discovery.path_patch_head(
                    full_ids, full_ids,
                    layer_idx, head_idx,
                    base_activations,
                    sft_activations
                )
                sft_prob = self.base_discovery._compute_answer_probability(
                    sft_patched_logits, target_tokens, answer_start_pos
                )
                sft_delta = sft_prob - base_prob

                # Patch RL activation into base model
                rl_patched_logits = self.base_discovery.path_patch_head(
                    full_ids, full_ids,
                    layer_idx, head_idx,
                    base_activations,
                    rl_activations
                )
                rl_prob = self.base_discovery._compute_answer_probability(
                    rl_patched_logits, target_tokens, answer_start_pos
                )
                rl_delta = rl_prob - base_prob

                sft_deltas.append(sft_delta)
                rl_deltas.append(rl_delta)

            # Aggregate across examples
            if sft_deltas:
                results['sft_deltas'].append(np.mean(sft_deltas))
                results['rl_deltas'].append(np.mean(rl_deltas))
                results['head_info'].append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'importance_score': score.score
                })

        return results

    def identify_vulnerable_circuits(
            self,
            cmap_results: Dict[str, List[float]],
            threshold: float = 0.1
    ) -> List[Dict]:
        """
        Identify heads that are more degraded under SFT than RL.
        These are the "vulnerable circuits" for regularization.

        Returns:
            List of vulnerable head information
        """
        vulnerable = []

        for i, head_info in enumerate(cmap_results['head_info']):
            sft_delta = cmap_results['sft_deltas'][i]
            rl_delta = cmap_results['rl_deltas'][i]

            # Vulnerable if SFT causes more negative change than RL
            if sft_delta < rl_delta - threshold:
                vulnerable.append({
                    **head_info,
                    'sft_delta': sft_delta,
                    'rl_delta': rl_delta,
                    'vulnerability': rl_delta - sft_delta
                })

        # Sort by vulnerability
        vulnerable.sort(key=lambda x: x['vulnerability'], reverse=True)

        print(f"\nIdentified {len(vulnerable)} vulnerable heads")
        if vulnerable:
            print("\nTop 5 most vulnerable heads:")
            for i, head in enumerate(vulnerable[:5]):
                print(f"  {i+1}. Layer {head['layer']}, Head {head['head']}: "
                      f"SFT Δ={head['sft_delta']:.4f}, RL Δ={head['rl_delta']:.4f}, "
                      f"Vulnerability={head['vulnerability']:.4f}")

        return vulnerable


def create_counterfactual_examples_math(dataset, n_examples: int = 100) -> List[Dict]:
    """
    Create meaningful counterfactuals for math problems by changing numbers.

    Strategy: Change numbers in the question to create a different problem
    with a different answer, while preserving the problem structure.

    Original: "What is 2 + 3?" (answer: 5)
    Counterfactual: "What is 5 + 6?" (answer: different, structure same)
    """
    import random

    examples = []

    for i in range(min(n_examples, len(dataset))):
        item = dataset[i]

        # Extract question and answer
        if isinstance(item, dict) and '0' in item:
            question = item['0'].get('value', '')
            try:
                answer = item['1']['ground_truth']['value']
            except (KeyError, TypeError):
                answer = str(item.get('1', ''))
        else:
            continue

        # Find all numbers in question
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)

        if len(numbers) >= 1:
            counterfactual = question

            # Change numbers with consistent offsets
            for num in numbers:
                try:
                    orig_num = float(num)
                    if orig_num < 10:
                        offset = random.randint(2, 5)
                    elif orig_num < 100:
                        offset = random.randint(5, 15)
                    else:
                        offset = random.randint(10, 30)

                    if '.' not in num:
                        new_num = str(int(orig_num + offset))
                    else:
                        new_num = str(round(orig_num + offset, 2))

                    counterfactual = counterfactual.replace(num, new_num, 1)
                except ValueError:
                    continue

            if counterfactual != question:
                examples.append({
                    'question': question,
                    'answer': str(answer),
                    'counterfactual_question': counterfactual,
                    'original_numbers': numbers
                })

        if len(examples) >= n_examples:
            break

    print(f"\n✅ Created {len(examples)} counterfactual pairs")
    if examples:
        print("\n📋 Sample counterfactuals:")
        for i, ex in enumerate(examples[:3]):
            print(f"  {i+1}. Original: {ex['question'][:80]}")
            print(f"     Counterfactual: {ex['counterfactual_question'][:80]}")
            print(f"     Answer: {ex['answer']}")

    return examples


def create_counterfactual_examples(dataset, n_examples: int = 100) -> List[Tuple[str, str]]:
    """Legacy function for backward compatibility."""
    examples = create_counterfactual_examples_math(dataset, n_examples)
    return [(ex['question'], ex['counterfactual_question']) for ex in examples]


def save_circuit_results(results: Dict, filepath: str):
    """Save circuit analysis results to JSON"""
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nCircuit analysis results saved to {filepath}")
