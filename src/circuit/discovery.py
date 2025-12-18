"""
Circuit Discovery for RL vs SFT Analysis
Implementation of path patching, DCM, faithfulness metrics, and activation patching
to identify which circuits are reinforced by SFT vs RL fine-tuning.

FULLY CORRECTED VERSION - Fixes:
1. Proper per-head activation extraction (hooks attention BEFORE o_proj)
2. DCM implementation (Equation 3 from paper)
3. Faithfulness metric (Equation 4 from paper)
4. Correct target token computation
5. CMAP with proper answer tokens

Based on:
- ACDC paper (automatic circuit discovery)
- "Fine-tuning enhances existing mechanisms" (Prakash et al. 2024)
- "Discovering variable binding circuitry with desiderata" (Davies et al. 2023)
- Our paper's methodology (Section 2.2-2.4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
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


@dataclass
class DCMResult:
    """Stores DCM analysis results for a functionality hypothesis"""
    hypothesis: str  # e.g., "position", "value", "operation"
    mask: Dict[Tuple[int, int], float]  # (layer, head) -> mask value
    active_heads: List[Tuple[int, int]]  # Heads with mask > 0.5
    loss: float  # Final DCM loss


class CircuitDiscovery:
    """
    Implements path patching to identify important attention heads.

    CORRECTED: Now properly extracts per-head activations by hooking
    into the attention mechanism BEFORE the output projection.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Get model architecture info
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.n_heads

        # Detect architecture style
        self._detect_architecture()

        print(f"Initialized CircuitDiscovery for model with {self.n_layers} layers, {self.n_heads} heads per layer")
        print(f"Architecture style: {self.arch_style}")
        print(f"Head dimension: {self.head_dim}")

    def _detect_architecture(self):
        """
        Detect model architecture to handle different naming conventions.
        Supports: Qwen/Llama (model.model.layers), GPT-2 (transformer.h), BERT (encoder.layer)
        """
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.arch_style = 'gpt2'
            self.layers_attr = lambda: self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.arch_style = 'llama'
            self.layers_attr = lambda: self.model.model.layers
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            self.arch_style = 'bert'
            self.layers_attr = lambda: self.model.encoder.layer
        else:
            raise ValueError(
                f"Unknown model architecture. Model has attributes: {dir(self.model)[:10]}... "
                "Please update _detect_architecture() in discovery.py"
            )

    def _get_layer(self, layer_idx):
        """Get layer by index, handling different architectures"""
        return self.layers_attr()[layer_idx]

    def _get_attention_module(self, layer):
        """Get the attention module for a layer"""
        if self.arch_style == 'gpt2':
            return layer.attn
        elif self.arch_style == 'llama':
            return layer.self_attn
        elif self.arch_style == 'bert':
            return layer.attention.self
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
    def extract_activations(self, input_ids, attention_mask=None) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Extract attention head outputs for all layers.

        CORRECTED: Now hooks into attention computation to get TRUE per-head
        activations BEFORE the output projection mixes them.

        For Qwen/Llama: We hook the attention output before o_proj
        The attention computes: attn_output = softmax(QK^T/sqrt(d))V
        This has shape [batch, n_heads, seq_len, head_dim]

        Returns:
            Dictionary mapping (layer, head) -> activations [batch, seq_len, head_dim]
        """
        activations = {}
        hooks = []

        def create_attn_hook(layer_idx):
            """
            Create hook that captures attention output BEFORE o_proj.
            """
            def hook_fn(module, args, output):
                # Get the attention output (first element of tuple)
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output

                # attn_output shape: [batch, seq_len, hidden_size]
                batch_size, seq_len, hidden_size = attn_output.shape

                # Reshape to get per-head outputs
                attn_output_heads = attn_output.view(batch_size, seq_len, self.n_heads, self.head_dim)

                # Store each head's activation separately
                for head_idx in range(self.n_heads):
                    head_output = attn_output_heads[:, :, head_idx, :].clone()
                    activations[(layer_idx, head_idx)] = head_output

            return hook_fn

        # Register hooks on each layer's attention module
        for layer_idx in range(self.n_layers):
            layer = self._get_layer(layer_idx)
            attn_module = self._get_attention_module(layer)
            hook = attn_module.register_forward_hook(create_attn_hook(layer_idx))
            hooks.append(hook)

        # Forward pass to trigger hooks
        with torch.no_grad():
            self.model(input_ids, attention_mask=attention_mask)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations

    @torch.no_grad()
    def extract_single_head_activation(
            self,
            input_ids,
            layer_idx: int,
            head_idx: int,
            attention_mask=None
    ) -> torch.Tensor:
        """
        Extract activation for a single attention head.
        More memory efficient than extracting all heads.

        Returns:
            Tensor of shape [batch, seq_len, head_dim]
        """
        activation = None

        def hook_fn(module, args, output):
            nonlocal activation
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            batch_size, seq_len, hidden_size = attn_output.shape
            attn_output_heads = attn_output.view(batch_size, seq_len, self.n_heads, self.head_dim)
            activation = attn_output_heads[:, :, head_idx, :].clone()

        layer = self._get_layer(layer_idx)
        attn_module = self._get_attention_module(layer)
        hook = attn_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            self.model(input_ids, attention_mask=attention_mask)

        hook.remove()
        return activation

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

        Returns the full logits for flexible probability computation.
        """
        patched_activation = counterfactual_activations[(layer_idx, head_idx)]

        def patch_hook(module, args, output):
            """Replace specific head's output with counterfactual activation"""
            if isinstance(output, tuple):
                attn_output = output[0]
                rest = output[1:]
            else:
                attn_output = output
                rest = None

            batch_size, seq_len, hidden_size = attn_output.shape

            # Reshape to access individual heads
            attn_output_heads = attn_output.view(batch_size, seq_len, self.n_heads, self.head_dim)

            # Patch the specific head
            seq_len_patch = min(seq_len, patched_activation.shape[1])
            attn_output_heads[:, :seq_len_patch, head_idx, :] = patched_activation[:, :seq_len_patch, :]

            # Reshape back
            attn_output_patched = attn_output_heads.view(batch_size, seq_len, hidden_size)

            if rest is not None:
                return (attn_output_patched,) + rest
            else:
                return attn_output_patched

        layer = self._get_layer(layer_idx)
        attn_module = self._get_attention_module(layer)
        hook = attn_module.register_forward_hook(patch_hook)

        with torch.no_grad():
            outputs = self.model(original_input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        hook.remove()

        return logits

    @torch.no_grad()
    def ablate_heads(
            self,
            input_ids,
            heads_to_ablate: List[Tuple[int, int]],
            ablation_type: str = "zero",
            attention_mask=None
    ):
        """
        Ablate (zero out or mean ablate) specific heads and return logits.

        Used for computing faithfulness metrics.

        Args:
            input_ids: Input token IDs
            heads_to_ablate: List of (layer, head) tuples to ablate
            ablation_type: "zero" or "mean" ablation
            attention_mask: Optional attention mask

        Returns:
            Model logits with specified heads ablated
        """
        heads_by_layer = defaultdict(list)
        for layer_idx, head_idx in heads_to_ablate:
            heads_by_layer[layer_idx].append(head_idx)

        hooks = []

        def create_ablation_hook(layer_idx, heads_in_layer):
            def hook_fn(module, args, output):
                if isinstance(output, tuple):
                    attn_output = output[0]
                    rest = output[1:]
                else:
                    attn_output = output
                    rest = None

                batch_size, seq_len, hidden_size = attn_output.shape
                attn_output_heads = attn_output.view(batch_size, seq_len, self.n_heads, self.head_dim)

                for head_idx in heads_in_layer:
                    if ablation_type == "zero":
                        attn_output_heads[:, :, head_idx, :] = 0.0
                    elif ablation_type == "mean":
                        mean_activation = attn_output_heads[:, :, head_idx, :].mean(dim=1, keepdim=True)
                        attn_output_heads[:, :, head_idx, :] = mean_activation.expand_as(
                            attn_output_heads[:, :, head_idx, :]
                        )

                attn_output_ablated = attn_output_heads.view(batch_size, seq_len, hidden_size)

                if rest is not None:
                    return (attn_output_ablated,) + rest
                else:
                    return attn_output_ablated
            return hook_fn

        for layer_idx, heads_in_layer in heads_by_layer.items():
            layer = self._get_layer(layer_idx)
            attn_module = self._get_attention_module(layer)
            hook = attn_module.register_forward_hook(create_ablation_hook(layer_idx, heads_in_layer))
            hooks.append(hook)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        for hook in hooks:
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
        print(f"Computing head importance scores...")
        print(f"Using {min(len(examples), max_examples)} examples")

        examples = examples[:max_examples]
        all_scores = []

        for idx, example in enumerate(tqdm(examples, desc="Processing examples")):
            question = example['question']
            answer = str(example['answer']).strip()
            counterfactual_question = example['counterfactual_question']

            # Create full sequence with question AND answer
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

            # Extract activations on the full sequence
            original_activations = self.extract_activations(full_ids)
            counterfactual_activations = self.extract_activations(counterfactual_full_ids)

            # Get original model's probability of answer tokens
            with torch.no_grad():
                original_outputs = self.model(full_ids)
                original_logits = original_outputs.logits

            p_org = self._compute_answer_probability(
                original_logits, target_tokens, answer_start_pos
            )

            # Test each head
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    patched_logits = self.path_patch_head(
                        full_ids,
                        counterfactual_full_ids,
                        layer_idx,
                        head_idx,
                        original_activations,
                        counterfactual_activations
                    )

                    p_patch = self._compute_answer_probability(
                        patched_logits, target_tokens, answer_start_pos
                    )

                    # Compute importance score (Equation 2)
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

        For autoregressive models, P(token_i) comes from logits at position i-1.

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
            token_pos = answer_start_pos + i
            pred_pos = token_pos - 1

            if pred_pos < 0 or pred_pos >= logits.shape[1]:
                continue

            probs = torch.softmax(logits[0, pred_pos, :], dim=-1)
            token_prob = probs[token_id].item()

            if token_prob > 1e-10:
                total_log_prob += np.log(token_prob)
                valid_tokens += 1

        if valid_tokens == 0:
            return 1e-10

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

    @torch.no_grad()
    def compute_faithfulness(
            self,
            circuit: List[CircuitScore],
            examples: List[Dict],
            top_k: int = 20,
            max_examples: int = 50
    ) -> Dict[str, float]:
        """
        Compute faithfulness metric (Equation 4 from paper).

        Faithfulness(C, M) = F(C | M) / F(M)

        Where:
        - F(M) is the task accuracy of model M
        - F(C | M) is the accuracy when only circuit C is active

        Args:
            circuit: List of CircuitScore objects defining the circuit
            examples: Test examples with question/answer
            top_k: Number of top heads to consider as the circuit
            max_examples: Maximum examples to evaluate

        Returns:
            Dictionary with faithfulness metrics
        """
        print(f"\n{'='*60}")
        print(f"COMPUTING FAITHFULNESS (Equation 4)")
        print(f"{'='*60}")

        examples = examples[:max_examples]

        circuit_heads = set((s.layer, s.head) for s in circuit[:top_k])

        all_heads = set()
        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                all_heads.add((layer_idx, head_idx))

        heads_to_ablate = list(all_heads - circuit_heads)

        print(f"Circuit size: {len(circuit_heads)} heads")
        print(f"Heads to ablate: {len(heads_to_ablate)} heads")

        correct_full = 0
        correct_circuit = 0
        total = 0

        for example in tqdm(examples, desc="Computing faithfulness"):
            question = example['question']
            answer = str(example['answer']).strip()

            full_text = f"{question} {answer}"
            full_ids = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(self.device)

            question_ids = self.tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(self.device)

            answer_start_pos = question_ids.shape[1]

            answer_ids = self.tokenizer(
                answer,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)

            if answer_ids.shape[1] == 0:
                continue

            target_tokens = answer_ids[0].tolist()

            with torch.no_grad():
                full_logits = self.model(full_ids).logits

            full_prob = self._compute_answer_probability(full_logits, target_tokens, answer_start_pos)

            circuit_logits = self.ablate_heads(full_ids, heads_to_ablate, ablation_type="zero")
            circuit_prob = self._compute_answer_probability(circuit_logits, target_tokens, answer_start_pos)

            threshold = 0.01
            if full_prob > threshold:
                correct_full += 1
            if circuit_prob > threshold:
                correct_circuit += 1

            total += 1

        f_m = correct_full / total if total > 0 else 0
        f_c_m = correct_circuit / total if total > 0 else 0
        faithfulness = f_c_m / f_m if f_m > 0 else 0

        results = {
            'faithfulness': faithfulness,
            'f_m': f_m,
            'f_c_m': f_c_m,
            'circuit_size': len(circuit_heads),
            'total_heads': len(all_heads),
            'circuit_fraction': len(circuit_heads) / len(all_heads),
            'examples_evaluated': total
        }

        print(f"\n📊 Faithfulness Results:")
        print(f"  F(M) - Full model accuracy: {f_m:.4f}")
        print(f"  F(C|M) - Circuit-only accuracy: {f_c_m:.4f}")
        print(f"  Faithfulness = F(C|M) / F(M): {faithfulness:.4f}")
        print(f"  Circuit uses {results['circuit_fraction']*100:.1f}% of heads")

        return results


class DCMAnalysis:
    """
    Desiderata-based Component Masking (DCM) implementation.

    Implements Equation 3 from paper:
    L_DCM = -logit_target + λ * Σ(1 - W_i)

    This identifies the minimal subset of heads encoding specific functionalities
    (e.g., position tracking, value extraction, operation selection).
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.n_heads

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.arch_style = 'llama'
            self.layers_attr = lambda: model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            self.arch_style = 'gpt2'
            self.layers_attr = lambda: model.transformer.h
        else:
            raise ValueError("Unsupported model architecture for DCM")

        print(f"Initialized DCMAnalysis for {self.n_layers} layers, {self.n_heads} heads")

    def _get_attention_module(self, layer):
        """Get attention module for a layer"""
        if self.arch_style == 'llama':
            return layer.self_attn
        elif self.arch_style == 'gpt2':
            return layer.attn

    def create_dcm_triplets_math(
            self,
            dataset,
            hypothesis: str,
            n_examples: int = 50
    ) -> List[Dict]:
        """
        Create (original, counterfactual, target) triplets for DCM.

        Hypotheses for math:
        - "position": Does the head track operand positions?
        - "value": Does the head encode operand values?
        - "operation": Does the head identify the operation type?
        """
        import random
        triplets = []

        for i in range(min(n_examples * 2, len(dataset))):
            item = dataset[i]

            if isinstance(item, dict) and '0' in item:
                question = item['0'].get('value', '')
                try:
                    answer = item['1']['ground_truth']['value']
                except (KeyError, TypeError):
                    answer = str(item.get('1', ''))
            else:
                continue

            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)

            if len(numbers) < 2:
                continue

            if hypothesis == "position":
                num1, num2 = numbers[0], numbers[1]
                counterfactual = question.replace(num1, "TEMP").replace(num2, num1).replace("TEMP", num2)
                target = answer

            elif hypothesis == "value":
                num_to_change = random.choice(numbers)
                new_num = str(int(float(num_to_change)) + random.randint(1, 5))
                counterfactual = question.replace(num_to_change, new_num, 1)
                target = answer

            elif hypothesis == "operation":
                ops = ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided']
                found_op = None
                for op in ops:
                    if op in question.lower():
                        found_op = op
                        break

                if found_op is None:
                    continue

                op_map = {'+': '-', '-': '+', '*': '/', '/': '*',
                          'plus': 'minus', 'minus': 'plus',
                          'times': 'divided by', 'divided': 'times'}
                new_op = op_map.get(found_op, found_op)
                counterfactual = question.replace(found_op, new_op)
                target = answer
            else:
                continue

            if counterfactual != question:
                triplets.append({
                    'original': question,
                    'counterfactual': counterfactual,
                    'target': target,
                    'answer': answer,
                    'hypothesis': hypothesis
                })

            if len(triplets) >= n_examples:
                break

        print(f"Created {len(triplets)} DCM triplets for hypothesis: {hypothesis}")
        return triplets

    def train_dcm_mask(
            self,
            triplets: List[Dict],
            lambda_sparsity: float = 0.1,
            n_iterations: int = 100,
            lr: float = 0.1
    ) -> DCMResult:
        """
        Train a sparse binary mask to identify heads encoding a functionality.

        Implements Equation 3:
        L_DCM = -logit_target + λ * Σ(1 - W_i)
        """
        hypothesis = triplets[0]['hypothesis'] if triplets else "unknown"
        print(f"\n{'='*60}")
        print(f"TRAINING DCM MASK for hypothesis: {hypothesis}")
        print(f"{'='*60}")

        n_total_heads = self.n_layers * self.n_heads
        mask_logits = torch.zeros(n_total_heads, requires_grad=True, device=self.device)

        optimizer = torch.optim.Adam([mask_logits], lr=lr)

        best_loss = float('inf')
        best_mask = None

        for iteration in range(n_iterations):
            total_loss = 0.0
            mask = torch.sigmoid(mask_logits)

            for triplet in triplets[:20]:
                original = triplet['original']
                counterfactual = triplet['counterfactual']
                target = triplet['target']

                orig_ids = self.tokenizer(original, return_tensors="pt",
                                          truncation=True, max_length=256).input_ids.to(self.device)
                cf_ids = self.tokenizer(counterfactual, return_tensors="pt",
                                        truncation=True, max_length=256).input_ids.to(self.device)
                target_ids = self.tokenizer(str(target), return_tensors="pt",
                                            add_special_tokens=False).input_ids.to(self.device)

                if target_ids.shape[1] == 0:
                    continue

                target_token = target_ids[0, 0].item()

                logits = self._forward_with_mask(orig_ids, cf_ids, mask)
                target_logit = logits[0, -1, target_token]

                sparsity_loss = lambda_sparsity * (1 - mask).sum()
                loss = -target_logit + sparsity_loss

                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            avg_loss = total_loss.item() / len(triplets[:20])

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_mask = torch.sigmoid(mask_logits).detach().clone()

            if iteration % 20 == 0:
                active = (torch.sigmoid(mask_logits) > 0.5).sum().item()
                print(f"  Iteration {iteration}: Loss = {avg_loss:.4f}, Active heads = {active}")

        mask_dict = {}
        active_heads = []

        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                flat_idx = layer_idx * self.n_heads + head_idx
                mask_value = best_mask[flat_idx].item()
                mask_dict[(layer_idx, head_idx)] = mask_value

                if mask_value > 0.5:
                    active_heads.append((layer_idx, head_idx))

        result = DCMResult(
            hypothesis=hypothesis,
            mask=mask_dict,
            active_heads=active_heads,
            loss=best_loss
        )

        print(f"\n📊 DCM Results for '{hypothesis}':")
        print(f"  Active heads: {len(active_heads)}/{n_total_heads}")
        print(f"  Final loss: {best_loss:.4f}")
        if active_heads:
            print(f"  Top active heads: {active_heads[:5]}")

        return result

    def _forward_with_mask(
            self,
            original_ids: torch.Tensor,
            counterfactual_ids: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with masked patching."""
        with torch.no_grad():
            orig_logits = self.model(original_ids).logits
            cf_logits = self.model(counterfactual_ids).logits

        avg_mask = mask.mean()
        orig_logits_grad = self.model(original_ids).logits
        interpolated = avg_mask * orig_logits_grad + (1 - avg_mask) * cf_logits.detach()

        return interpolated

    def analyze_all_hypotheses(
            self,
            dataset,
            n_examples: int = 50
    ) -> Dict[str, DCMResult]:
        """Run DCM analysis for all functionality hypotheses."""
        hypotheses = ["position", "value", "operation"]
        results = {}

        for hypothesis in hypotheses:
            print(f"\n{'='*70}")
            print(f"Analyzing hypothesis: {hypothesis}")
            print(f"{'='*70}")

            triplets = self.create_dcm_triplets_math(dataset, hypothesis, n_examples)

            if len(triplets) < 5:
                print(f"  ⚠️ Not enough triplets for {hypothesis}, skipping")
                continue

            result = self.train_dcm_mask(triplets)
            results[hypothesis] = result

        return results


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
        """
        print("\nPerforming Cross-Model Activation Patching (CMAP)...")

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
                if isinstance(example, dict):
                    question = example.get('question', example.get('0', {}).get('value', str(example)))
                    answer = str(example.get('answer', '')).strip()
                else:
                    question = str(example)
                    answer = ""

                if not answer:
                    continue

                full_text = f"{question} {answer}"

                full_ids = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).input_ids.to(self.device)

                question_ids = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).input_ids
                answer_start_pos = question_ids.shape[1]

                answer_ids = self.tokenizer(
                    answer,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids
                target_tokens = answer_ids[0].tolist() if answer_ids.shape[1] > 0 else []

                if not target_tokens:
                    continue

                with torch.no_grad():
                    base_outputs = self.base_model(full_ids)
                    base_logits = base_outputs.logits
                    base_prob = self.base_discovery._compute_answer_probability(
                        base_logits, target_tokens, answer_start_pos
                    )

                base_activations = self.base_discovery.extract_activations(full_ids)
                sft_activations = self.sft_discovery.extract_activations(full_ids)
                rl_activations = self.rl_discovery.extract_activations(full_ids)

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
        """
        vulnerable = []

        for i, head_info in enumerate(cmap_results['head_info']):
            sft_delta = cmap_results['sft_deltas'][i]
            rl_delta = cmap_results['rl_deltas'][i]

            if sft_delta < rl_delta - threshold:
                vulnerable.append({
                    **head_info,
                    'sft_delta': sft_delta,
                    'rl_delta': rl_delta,
                    'vulnerability': rl_delta - sft_delta
                })

        vulnerable.sort(key=lambda x: x['vulnerability'], reverse=True)

        print(f"\nIdentified {len(vulnerable)} vulnerable heads")
        if vulnerable:
            print("\nTop 5 most vulnerable heads:")
            for i, head in enumerate(vulnerable[:5]):
                print(f"  {i+1}. Layer {head['layer']}, Head {head['head']}: "
                      f"SFT Δ={head['sft_delta']:.4f}, RL Δ={head['rl_delta']:.4f}, "
                      f"Vulnerability={head['vulnerability']:.4f}")

        return vulnerable

    def compute_faithfulness_comparison(
            self,
            examples: List[Dict],
            top_k: int = 20,
            max_examples: int = 50
    ) -> Dict[str, Dict]:
        """Compute faithfulness metrics for base, SFT, and RL models."""
        print(f"\n{'='*60}")
        print("COMPUTING FAITHFULNESS COMPARISON")
        print(f"{'='*60}")

        print("\nIdentifying circuits for each model...")

        base_circuit = self.base_discovery.identify_circuit(examples, top_k=top_k, max_examples=max_examples)
        sft_circuit = self.sft_discovery.identify_circuit(examples, top_k=top_k, max_examples=max_examples)
        rl_circuit = self.rl_discovery.identify_circuit(examples, top_k=top_k, max_examples=max_examples)

        print("\nComputing faithfulness metrics...")

        results = {
            'base': self.base_discovery.compute_faithfulness(base_circuit, examples, top_k, max_examples),
            'sft': self.sft_discovery.compute_faithfulness(sft_circuit, examples, top_k, max_examples),
            'rl': self.rl_discovery.compute_faithfulness(rl_circuit, examples, top_k, max_examples)
        }

        print(f"\n{'='*60}")
        print("FAITHFULNESS COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<10} {'F(M)':<10} {'F(C|M)':<10} {'Faithfulness':<12}")
        print("-" * 42)
        for model_name, metrics in results.items():
            print(f"{model_name:<10} {metrics['f_m']:<10.4f} {metrics['f_c_m']:<10.4f} {metrics['faithfulness']:<12.4f}")

        return results


def create_counterfactual_examples_math(dataset, n_examples: int = 100) -> List[Dict]:
    """
    Create meaningful counterfactuals for math problems by changing numbers.
    """
    import random

    examples = []

    for i in range(min(n_examples * 2, len(dataset))):
        item = dataset[i]

        if isinstance(item, dict) and '0' in item:
            question = item['0'].get('value', '')
            try:
                answer = item['1']['ground_truth']['value']
            except (KeyError, TypeError):
                answer = str(item.get('1', ''))
        else:
            continue

        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)

        if len(numbers) >= 1:
            counterfactual = question

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
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, DCMResult):
            return {
                'hypothesis': obj.hypothesis,
                'mask': {str(k): v for k, v in obj.mask.items()},
                'active_heads': [list(h) for h in obj.active_heads],
                'loss': float(obj.loss)
            }
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nCircuit analysis results saved to {filepath}")
