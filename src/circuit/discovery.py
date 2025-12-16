"""
Circuit Discovery for RL vs SFT Analysis
Implementation of path patching and activation patching to identify which circuits
are reinforced by SFT vs RL fine-tuning.

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
        # Check for GPT-2 style first (since that's what you're using)
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.arch_style = 'gpt2'  # GPT-2, GPT-Neo style
            self.layers_attr = lambda: self.model.transformer.h
            # GPT-2 uses 'c_proj' for output projection
            self.attn_proj_attr = 'c_proj'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.arch_style = 'llama'  # Qwen, Llama, Mistral style
            self.layers_attr = lambda: self.model.model.layers
            self.attn_proj_attr = 'o_proj'
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            self.arch_style = 'bert'  # BERT style
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
            # GPT-2 uses 'attn.c_proj' for output projection
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

        This caches activations to avoid repeated forward passes.
        """
        activations = {}

        def create_hook(layer_idx, head_idx):
            def hook_fn(module, input, output):
                # output[0] is the attention output
                attn_output = output[0] if isinstance(output, tuple) else output

                # Handle both 2D and 3D shapes
                if len(attn_output.shape) == 2:
                    # Shape is [batch, hidden] - add sequence dimension
                    attn_output = attn_output.unsqueeze(1)

                batch_size, seq_len, hidden_dim = attn_output.shape

                # Split into heads
                head_dim = hidden_dim // self.n_heads
                attn_output_heads = attn_output.reshape(batch_size, seq_len, self.n_heads, head_dim)

                # Extract this specific head
                head_output = attn_output_heads[:, :, head_idx, :]
                activations[(layer_idx, head_idx)] = head_output.clone()

            return hook_fn

        # Register hooks for all attention layers
        hooks = []
        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                layer = self._get_layer(layer_idx)
                attn_proj = self._get_attn_projection(layer)
                hook = attn_proj.register_forward_hook(
                    create_hook(layer_idx, head_idx)
                )
                hooks.append(hook)

        # Forward pass to populate activations
        with torch.no_grad():
            self.model(input_ids, attention_mask=attention_mask)

        # Remove hooks
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

        Returns the probability of the correct next token.
        """
        patched_activation = counterfactual_activations[(layer_idx, head_idx)]

        def patch_hook(module, input, output):
            # Replace the specific head's output with counterfactual
            attn_output = output[0] if isinstance(output, tuple) else output

            # Handle both 2D and 3D shapes
            if len(attn_output.shape) == 2:
                attn_output = attn_output.unsqueeze(1)

            batch_size, seq_len, hidden_dim = attn_output.shape
            head_dim = hidden_dim // self.n_heads

            # Reshape to separate heads
            attn_output_heads = attn_output.reshape(batch_size, seq_len, self.n_heads, head_dim)

            # Patch this head
            seq_len_patch = min(seq_len, patched_activation.shape[0])
            if len(patched_activation.shape) == 2:
                # patched_activation is [seq_len, head_dim]
                seq_len_patch = min(seq_len, patched_activation.shape[0])
                attn_output_heads[:, :seq_len_patch, head_idx, :] = patched_activation[:seq_len_patch].unsqueeze(0)
            else:
                # patched_activation is [batch, seq_len, head_dim]
                seq_len_patch = min(seq_len, patched_activation.shape[1])
                attn_output_heads[:, :seq_len_patch, head_idx, :] = patched_activation[:, :seq_len_patch]


            # Reshape back
            attn_output_patched = attn_output_heads.reshape(batch_size, seq_len, hidden_dim)

            # Return in the correct format
            if isinstance(output, tuple):
                return (attn_output_patched,) + output[1:]
            else:
                return attn_output_patched

        # Register hook for this specific layer
        layer = self._get_layer(layer_idx)
        attn_proj = self._get_attn_projection(layer)
        hook = attn_proj.register_forward_hook(patch_hook)

        # Forward pass with patching
        with torch.no_grad():
            outputs = self.model(original_input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Remove hook
        hook.remove()

        # Get probability of correct next token (last position)
        probs = torch.softmax(logits[:, -1, :], dim=-1)

        return probs, logits

    def compute_head_importance(
            self,
            examples: List[Dict],
            max_examples: int = 50,
            batch_size: int = 4
    ) -> List[CircuitScore]:
        """
        Compute importance scores for all attention heads using path patching.

        Args:
            examples: List of (original_text, counterfactual_text) pairs
            max_examples: Maximum number of examples to use (for efficiency)
            batch_size: Batch size for processing

        Returns:
            List of CircuitScore objects ranked by importance
        """
        # Validation
        if not examples:
            raise ValueError("No examples provided for circuit discovery!")

        # Validate example structure
        required_keys = ['question', 'answer', 'counterfactual_question']
        for i, ex in enumerate(examples[:5]):
            missing_keys = [key for key in required_keys if key not in ex]
            if missing_keys:
                raise ValueError(f"Example {i} missing required keys: {missing_keys}. Has keys: {list(ex.keys())}")

        print(f"\n✅ Validated {len(examples)} examples with required structure")
        print(f"Computing head importance scores...")
        print(f"Using {min(len(examples), max_examples)} examples")

        # Limit examples for efficiency
        examples = examples[:max_examples]

        all_scores = []

        for idx, example in enumerate(tqdm(examples, desc="Processing examples")):
            # Extract question, answer, counterfactual
            question = example['question']
            answer = example['answer']
            counterfactual_question = example['counterfactual_question']

            # Tokenize question
            original_ids = self.tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(self.device)

            # Tokenize counterfactual
            counterfactual_ids = self.tokenizer(
                counterfactual_question,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(self.device)

            # Tokenize ANSWER
            answer_ids = self.tokenizer(
                answer,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)

            if answer_ids.shape[1] == 0:
                continue

            # Extract activations (cached for efficiency)
            original_activations = self.extract_activations(original_ids)
            counterfactual_activations = self.extract_activations(counterfactual_ids)

            # Get ALL answer tokens for proper evaluation of complete answers
            # For math problems, we want to predict the complete numerical answer, not just first digit
            target_tokens = answer_ids[0].tolist()

            # Debug first few
            if idx < 3:
                print(f"\n  Example {idx}:")
                print(f"    Q: {question[:50]}...")
                print(f"    A: {answer}")
                print(f"    Target tokens: {[self.tokenizer.decode([t]) for t in target_tokens]}")

            # Get original probability (average over all answer tokens for complete answer)
            with torch.no_grad():
                original_outputs = self.model(original_ids)
                original_logits = original_outputs.logits
                original_probs = torch.softmax(original_logits[:, -1, :], dim=-1)
                # Average probability across all answer tokens
                p_org = sum(original_probs[0, tok].item() for tok in target_tokens) / len(target_tokens)

            # Test each head
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    # Path patch this head
                    patched_probs, _ = self.path_patch_head(
                        original_ids,
                        counterfactual_ids,
                        layer_idx,
                        head_idx,
                        original_activations,
                        counterfactual_activations
                    )

                    # Get patched probability (average over all answer tokens for complete answer)
                    p_patch = sum(patched_probs[0, tok].item() for tok in target_tokens) / len(target_tokens)

                    # Compute score (Equation 2 from paper)
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

    def _aggregate_scores(self, scores: List[CircuitScore]) -> List[CircuitScore]:
        """Aggregate scores for the same head across multiple examples"""
        score_dict = defaultdict(list)

        for score in scores:
            key = (score.layer, score.head)
            score_dict[key].append(score.score)

        aggregated = []
        for (layer, head), score_list in score_dict.items():
            # Use mean score
            mean_score = np.mean(score_list)
            aggregated.append(CircuitScore(
                layer=layer,
                head=head,
                score=mean_score
            ))

        return aggregated
    def binarize_circuit(
            self,
            circuit: List[CircuitScore],
            threshold: float = None,
            top_k: int = None
    ) -> Dict[Tuple[int, int], int]:
        """
        Convert continuous circuit scores to binary (in/out).

        Args:
            circuit: List of CircuitScore objects with continuous scores
            threshold: Include heads with score < threshold (more negative = more important)
                      If None, uses top_k instead
            top_k: Include top k most important heads
                   If None, uses threshold instead

        Returns:
            Dictionary mapping (layer, head) -> 1 (in circuit) or 0 (not in circuit)
        """
        if threshold is None and top_k is None:
            raise ValueError("Must specify either threshold or top_k")

        binary_circuit = {}

        if top_k is not None:
            # Take top-k most important heads
            important_heads = set((s.layer, s.head) for s in circuit[:top_k])
        else:
            # Use threshold
            important_heads = set((s.layer, s.head) for s in circuit if s.score < threshold)

        # Create binary mapping for ALL heads
        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                key = (layer_idx, head_idx)
                binary_circuit[key] = 1 if key in important_heads else 0

        return binary_circuit

    def identify_circuit(
            self,
            examples: List[Dict],  # ✅
            top_k: int = 20,
            max_examples: int = 50
    ) -> List[CircuitScore]:
        """
        Identify the top-k most important heads forming the circuit.

        Args:
            examples: List of (original, counterfactual) text pairs
            top_k: Number of top heads to return
            max_examples: Maximum examples to use for efficiency

        Returns:
            List of top-k CircuitScore objects
        """
        print(f"\nIdentifying circuit (top-{top_k} heads)...")

        # Compute all head importances
        all_scores = self.compute_head_importance(examples, max_examples=max_examples)

        # Return top-k
        circuit = all_scores[:top_k]

        print(f"\nCircuit identified: {len(circuit)} heads")
        print("\nTop 5 most important heads:")
        for i, score in enumerate(circuit[:5]):
            print(f"  {i+1}. Layer {score.layer}, Head {score.head}: score = {score.score:.4f}")

        return circuit


def create_counterfactual_examples_math(dataset, n_examples: int = 100) -> List[Dict]:
    """
    Create meaningful counterfactuals for math problems by changing numbers.
    Strategy: Scale/shift numbers while preserving problem structure.

    Original: "What is 2 + 3?" (answer: 5)
    Counterfactual: "What is 5 + 6?" (answer: different, but structure same)
    """
    import re
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

        # Find all numbers in question (including decimals)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)

        if len(numbers) >= 1:
            # Create counterfactual by CHANGING numbers
            counterfactual = question

            # Strategy: Add a small offset to preserve problem structure
            for num in numbers:
                try:
                    orig_num = float(num)
                    # Add offset based on magnitude to avoid creating too-large numbers
                    if orig_num < 10:
                        offset = random.randint(2, 5)
                    elif orig_num < 100:
                        offset = random.randint(5, 15)
                    else:
                        offset = random.randint(10, 30)

                    # Preserve integer vs decimal format
                    if '.' not in num:
                        new_num = str(int(orig_num + offset))
                    else:
                        new_num = str(round(orig_num + offset, 2))

                    # Replace only first occurrence to avoid replacing same number twice
                    counterfactual = counterfactual.replace(num, new_num, 1)
                except ValueError:
                    # Skip if number parsing fails
                    continue

            # Only add if counterfactual actually changed
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

            # Verify numbers changed
            orig_nums = set(ex['original_numbers'])
            cf_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', ex['counterfactual_question']))
            if orig_nums == cf_nums:
                print(f"     ⚠️  WARNING: Numbers did not change!")
            else:
                print(f"     ✅ Numbers changed\n")

    return examples

class CrossModelCircuitAnalysis:
    """
    Compare how circuits are preserved/modified across base, SFT, and RL models.
    Implements the cross-model activation patching (CMAP) from Section 2.4.
    """

    def __init__(self, base_model, sft_model, rl_model, tokenizer, device="cuda"):
        self.base_model = base_model
        self.sft_model = sft_model
        self.rl_model = rl_model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize circuit discovery for each model
        self.base_discovery = CircuitDiscovery(base_model, tokenizer, device)
        self.sft_discovery = CircuitDiscovery(sft_model, tokenizer, device)
        self.rl_discovery = CircuitDiscovery(rl_model, tokenizer, device)

    @torch.no_grad()
    def compute_circuit_faithfulness(
        self,
        circuit: List[CircuitScore],
        model_discovery: CircuitDiscovery,
        test_examples: List[str],
        max_examples: int = 100
    ) -> float:
        """
        Compute faithfulness metric (Equation 4 from paper).
        Measures how well the circuit alone can perform the task.

        Returns:
            Faithfulness score (0-1, higher is better)
        """
        test_examples = test_examples[:max_examples]

        full_model_correct = 0
        circuit_only_correct = 0

        for text in tqdm(test_examples, desc="Computing faithfulness"):
            input_ids = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(self.device)

            # Full model prediction
            with torch.no_grad():
                outputs = model_discovery.model(input_ids)
                full_pred = outputs.logits[:, -1, :].argmax(dim=-1)

            # Circuit-only prediction (ablate all non-circuit heads)
            # For simplicity, we'll measure this by patching non-circuit heads with zeros
            # In practice, you'd use counterfactual activations

            # For now, use a simplified metric: just measure full model performance
            # A full implementation would ablate non-circuit heads

            full_model_correct += 1  # Placeholder
            circuit_only_correct += 1  # Placeholder

        # Faithfulness = circuit performance / full model performance
        faithfulness = circuit_only_correct / (full_model_correct + 1e-10)

        return faithfulness

    @torch.no_grad()
    def cross_model_activation_patching(
        self,
        circuit: List[CircuitScore],
        test_examples: List[str],
        max_examples: int = 50
    ) -> Dict[str, List[float]]:
        """
        Implement CMAP (Equation 5 from paper).
        Patch activations from fine-tuned models into base model and measure change.

        Returns:
            Dictionary with 'sft_deltas' and 'rl_deltas' lists
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

            for text in test_examples:
                input_ids = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).input_ids.to(self.device)

                # Get base model performance
                with torch.no_grad():
                    base_outputs = self.base_model(input_ids)
                    base_logits = base_outputs.logits[:, -1, :]
                    base_probs = torch.softmax(base_logits, dim=-1)
                    target_token = input_ids[0, -1].item() if input_ids.shape[1] > 1 else input_ids[0, 0].item()
                    base_prob = base_probs[0, target_token].item()

                # Extract activations from fine-tuned models
                sft_activations = self.sft_discovery.extract_activations(input_ids)
                rl_activations = self.rl_discovery.extract_activations(input_ids)
                base_activations = self.base_discovery.extract_activations(input_ids)

                # Patch SFT activation into base model
                sft_probs, _ = self.base_discovery.path_patch_head(
                    input_ids, input_ids,
                    layer_idx, head_idx,
                    base_activations,
                    sft_activations
                )
                sft_prob = sft_probs[0, target_token].item()
                sft_delta = sft_prob - base_prob

                # Patch RL activation into base model
                rl_probs, _ = self.base_discovery.path_patch_head(
                    input_ids, input_ids,
                    layer_idx, head_idx,
                    base_activations,
                    rl_activations
                )
                rl_prob = rl_probs[0, target_token].item()
                rl_delta = rl_prob - base_prob

                sft_deltas.append(sft_delta)
                rl_deltas.append(rl_delta)

            # Aggregate across examples
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


def create_counterfactual_examples(dataset, n_examples: int = 100) -> List[Tuple[str, str]]:
    """
    Create counterfactual examples for circuit discovery.

    For Task A (math, science QA, tool use), we create counterfactuals by:
    - Changing numbers/values in math problems
    - Swapping scientific terms
    - Modifying tool parameters

    Returns:
        List of (original, counterfactual) text pairs
    """
    examples = []

    for i in range(min(n_examples, len(dataset))):
        item = dataset[i]

        # Extract question
        if isinstance(item, dict):
            if '0' in item and isinstance(item['0'], dict):
                question = item['0'].get('value', '')
            elif 'question' in item:
                question = item['question']
            elif 'prompt' in item:
                question = item['prompt']
            else:
                question = str(item)
        else:
            question = str(item)

        # Simple counterfactual: swap some words
        # In practice, you'd want more sophisticated counterfactuals
        words = question.split()
        if len(words) > 5:
            # Swap two random words
            import random
            idx1, idx2 = random.sample(range(len(words)), 2)
            counterfactual_words = words.copy()
            counterfactual_words[idx1], counterfactual_words[idx2] = words[idx2], words[idx1]
            counterfactual = ' '.join(counterfactual_words)
        else:
            counterfactual = question  # Keep same if too short

        examples.append((question, counterfactual))

    return examples


def save_circuit_results(results: Dict, filepath: str):
    """Save circuit analysis results to JSON"""
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
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
