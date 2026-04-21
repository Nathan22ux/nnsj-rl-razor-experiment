"""
Circuit Discovery for RL vs SFT Analysis.

Uses Differential Binary Masking (DBM) to identify which attention heads
are active for a given task, then compares circuits across base, SFT, and
RL fine-tuned models.

DBM reference: Chaudhary & Geiger (2024), arxiv 2409.04478
    interpolated = (1 - sigma(m/T)) * f_base + sigma(m/T) * f_source
    L = CE(model(interpolated), y)
    T annealed 10->0.1 over 20 epochs
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


class DCMAnalysis:
    """
    Desiderata-based Component Masking (DCM) implementation.

    Implements Equation 3 from paper:
    L_DCM = -logit_target + λ * Σ(1 - W_i)

    This identifies the minimal subset of heads encoding specific functionalities
    (e.g., position tracking, value extraction, operation selection).

    FIXED: Now uses proper per-head activation-level masking.
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

    def create_dcm_triplets_science(
            self,
            dataset,
            hypothesis: str,
            n_examples: int = 500
    ) -> List[Dict]:
        """
        Create (original, counterfactual, target) triplets for DCM on the
        SciKnowEval chemistry dataset.

        Handles all 6 task types:
          MCQ  — molar_weight_calculation, molecular_property_calculation,
                 molecule_structure_prediction, reaction_prediction, retrosynthesis
          Fill — balancing_chemical_equation

        Hypotheses:
          "answer_key"  — Does the head track which letter (A/B/C/D) is correct?
                          Counterfactual: rotate all choices left by 1 position so
                          the correct answer shifts to a different letter.
          "molecule"    — Does the head encode molecule/compound identity?
                          Counterfactual: swap with a different item from the same
                          task type, keeping question structure identical.
          "task_type"   — Does the head identify the chemistry sub-task?
                          Counterfactual: replace with an MCQ from a different task
                          type that shares the same answer key.
        """
        import random

        triplets = []
        items = list(dataset)

        def format_mcq(item):
            """Append A/B/C/D choices to the question text."""
            q = item.get('question', '')
            labels = item.get('choices', {}).get('label', [])
            texts = item.get('choices', {}).get('text', [])
            if labels and texts:
                opts = "\n".join(f"{l}: {t}" for l, t in zip(labels, texts))
                return f"{q}\n{opts}"
            return q

        # Split by question type
        mcq_items = [x for x in items if x.get('type') == 'mcq-4-choices'
                     and x.get('answerKey') and x.get('choices', {}).get('text')]
        fill_items = [x for x in items if x.get('type') == 'filling'
                      and str(x.get('answer', '')).strip()]

        # Group MCQ by task for task_type hypothesis
        task_groups: Dict[str, List] = {}
        for x in mcq_items:
            task = x.get('details', {}).get('task', 'unknown')
            task_groups.setdefault(task, []).append(x)

        if hypothesis == "answer_key":
            # Rotate MCQ choices left by 1: [B,C,D,A].
            # Correct letter moves one position earlier: A→D, B→A, C→B, D→C.
            labels_order = ["A", "B", "C", "D"]
            for item in mcq_items:
                texts = item['choices']['text']       # [opt_A, opt_B, opt_C, opt_D]
                answer_key = item['answerKey']
                if answer_key not in labels_order or len(texts) != 4:
                    continue

                correct_idx = labels_order.index(answer_key)
                rotated_texts = texts[1:] + texts[:1]          # rotate left
                new_correct_idx = (correct_idx - 1) % 4
                new_correct_key = labels_order[new_correct_idx]

                orig_opts = "\n".join(f"{l}: {t}" for l, t in zip(labels_order, texts))
                cf_opts   = "\n".join(f"{l}: {t}" for l, t in zip(labels_order, rotated_texts))

                triplets.append({
                    'original':       f"{item['question']}\n{orig_opts}",
                    'counterfactual': f"{item['question']}\n{cf_opts}",
                    'target':         answer_key,
                    'answer':         answer_key,
                    'hypothesis':     hypothesis,
                })
                if len(triplets) >= n_examples:
                    break

        elif hypothesis == "molecule":
            # Swap with a different item from the same task type (different molecule).
            # For filling: use the equation text as the "molecule" feature.
            source_items = mcq_items + fill_items
            task_all: Dict[str, List] = {}
            for x in source_items:
                t = x.get('details', {}).get('task', 'unknown')
                task_all.setdefault(t, []).append(x)

            for item in source_items:
                task = item.get('details', {}).get('task', 'unknown')
                same_task = task_all.get(task, [])
                candidates = [x for x in same_task if x is not item]
                if not candidates:
                    continue

                cf_item = random.choice(candidates)
                item_type = item.get('type', '')

                if item_type == 'mcq-4-choices':
                    original_q = format_mcq(item)
                    cf_q       = format_mcq(cf_item)
                    target     = item['answerKey']
                else:  # filling
                    original_q = item['question']
                    cf_q       = cf_item['question']
                    target     = str(item.get('answer', '')).strip()

                if not target:
                    continue

                triplets.append({
                    'original':       original_q,
                    'counterfactual': cf_q,
                    'target':         target,
                    'answer':         target,
                    'hypothesis':     hypothesis,
                })
                if len(triplets) >= n_examples:
                    break

        elif hypothesis == "task_type":
            # MCQ from task A paired with MCQ from a different task B,
            # preferring the same answer key so the target token is consistent.
            task_names = list(task_groups.keys())
            if len(task_names) < 2:
                print("  Not enough task types for task_type hypothesis")
                return triplets

            for item in mcq_items:
                task       = item.get('details', {}).get('task', 'unknown')
                answer_key = item.get('answerKey', 'A')
                other_tasks = [t for t in task_names if t != task]
                cf_task    = random.choice(other_tasks)

                # Prefer same answer key for a clean counterfactual
                same_key = [x for x in task_groups[cf_task] if x.get('answerKey') == answer_key]
                cf_item   = random.choice(same_key) if same_key else random.choice(task_groups[cf_task])

                triplets.append({
                    'original':       format_mcq(item),
                    'counterfactual': format_mcq(cf_item),
                    'target':         answer_key,
                    'answer':         answer_key,
                    'hypothesis':     hypothesis,
                })
                if len(triplets) >= n_examples:
                    break

        print(f"Created {len(triplets)} DCM triplets for hypothesis: {hypothesis}")
        return triplets

    def train_dcm_mask(
            self,
            triplets: List[Dict],
            n_epochs: int = 20,
            lr: float = 0.001,
            batch_size: int = 16,
            temp_start: float = 10.0,
            temp_end: float = 0.1,
    ) -> DCMResult:
        """
        Train a Differential Binary Mask (DBM) to identify heads encoding a functionality.

        Implements the DBM method from Chaudhary & Geiger (2024), arxiv 2409.04478:

            interpolated = (1 - sigma(m/T)) odot f_base + sigma(m/T) odot f_source
            L = CE(model(interpolated), y)

        Temperature T is annealed linearly from temp_start to temp_end over
        n_epochs, pushing sigma(m/T) toward 0 or 1 (binary) by the final epoch.
        After training, heads where sigma(m/T_final) > 0.5 are considered active.
        """
        hypothesis = triplets[0]['hypothesis'] if triplets else "unknown"
        print(f"\n{'='*60}")
        print(f"TRAINING DBM for hypothesis: {hypothesis}")
        print(f"  Epochs={n_epochs}, lr={lr}, batch={batch_size}, T: {temp_start}→{temp_end}")
        print(f"{'='*60}")

        n_total_heads = self.n_layers * self.n_heads
        # Initialise mask logits to 0 → sigma(0/T)=0.5 (neutral start)
        mask_logits = torch.zeros(n_total_heads, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([mask_logits], lr=lr)

        best_loss = float('inf')
        best_mask_logits = mask_logits.detach().clone()

        import random as _random
        for epoch in range(n_epochs):
            # Linear temperature annealing
            T = temp_start + (temp_end - temp_start) * epoch / max(n_epochs - 1, 1)

            epoch_loss = 0.0
            valid = 0

            batch = _random.sample(triplets, min(batch_size, len(triplets)))

            for triplet in batch:
                orig_ids = self.tokenizer(
                    triplet['original'], return_tensors="pt",
                    truncation=True, max_length=256
                ).input_ids.to(self.device)
                cf_ids = self.tokenizer(
                    triplet['counterfactual'], return_tensors="pt",
                    truncation=True, max_length=256
                ).input_ids.to(self.device)
                target_ids = self.tokenizer(
                    str(triplet['target']), return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(self.device)

                if target_ids.shape[1] == 0:
                    continue

                target_token = target_ids[0, 0].item()

                # DBM mask: sigma(m/T)  — sharpens toward 0/1 as T→0
                mask = torch.sigmoid(mask_logits / T)

                logits = self._forward_with_dbm(orig_ids, cf_ids, mask)

                # CE loss on target token at last position
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                loss = -log_probs[target_token]

                epoch_loss += loss
                valid += 1

            if valid == 0:
                continue

            avg_loss = epoch_loss / valid
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_([mask_logits], max_norm=1.0)
            optimizer.step()

            loss_val = avg_loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_mask_logits = mask_logits.detach().clone()

            active = (torch.sigmoid(mask_logits / T) > 0.5).sum().item()
            print(f"  Epoch {epoch+1:2d}/{n_epochs}: T={T:.2f}  loss={loss_val:.4f}  active={active}")

        # Read off final binary mask at T_final
        final_mask = torch.sigmoid(best_mask_logits / temp_end)

        mask_dict = {}
        active_heads = []
        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                flat_idx = layer_idx * self.n_heads + head_idx
                mask_value = final_mask[flat_idx].item()
                mask_dict[(layer_idx, head_idx)] = mask_value
                if mask_value > 0.5:
                    active_heads.append((layer_idx, head_idx))

        result = DCMResult(
            hypothesis=hypothesis,
            mask=mask_dict,
            active_heads=active_heads,
            loss=best_loss
        )

        print(f"\n📊 DBM Results for '{hypothesis}':")
        print(f"  Active heads: {len(active_heads)}/{n_total_heads}")
        print(f"  Final loss: {best_loss:.4f}")
        if active_heads:
            print(f"  Top active heads: {active_heads[:5]}")

        return result

    def _forward_with_dbm(
            self,
            base_ids: torch.Tensor,
            source_ids: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using DBM interpolation (Chaudhary & Geiger 2024):

            h = (1 - mask[head]) * f_base + mask[head] * f_source

        mask[head] = sigma(m/T):
          - mask → 0  →  use base activations  (head is OFF / not selected)
          - mask → 1  →  use source activations (head is ON  / selected)

        Runs the base forward pass; at each o_proj input, swaps in source
        activations for heads where mask is high.
        """
        hooks = []
        source_activations = {}

        # Step 1: capture source (counterfactual) activations — no grad needed
        def capture_source_hook(layer_idx):
            def hook_fn(module, args):
                if len(args) > 0:
                    o_proj_input = args[0]
                    bs, seq, hidden = o_proj_input.shape
                    heads = o_proj_input.view(bs, seq, self.n_heads, self.head_dim)
                    for head_idx in range(self.n_heads):
                        source_activations[(layer_idx, head_idx)] = heads[:, :, head_idx, :].clone()
                return args
            return hook_fn

        for layer_idx in range(self.n_layers):
            layer = self.layers_attr()[layer_idx]
            attn = self._get_attention_module(layer)
            proj = attn.o_proj if hasattr(attn, 'o_proj') else attn.c_proj
            hooks.append(proj.register_forward_pre_hook(capture_source_hook(layer_idx)))

        with torch.no_grad():
            self.model(source_ids)

        for h in hooks:
            h.remove()
        hooks = []

        # Step 2: run base forward, interpolating with source using DBM mask
        def create_dbm_hook(layer_idx):
            def hook_fn(module, args):
                if not args:
                    return args
                o_proj_input = args[0]
                bs, seq, hidden = o_proj_input.shape
                base_heads = o_proj_input.view(bs, seq, self.n_heads, self.head_dim)
                out_heads = base_heads.clone()

                for head_idx in range(self.n_heads):
                    flat_idx = layer_idx * self.n_heads + head_idx
                    m = mask[flat_idx]  # sigma(logit/T), gradient flows through this

                    src = source_activations.get((layer_idx, head_idx))
                    if src is not None:
                        seq_len = min(seq, src.shape[1])
                        # DBM formula: (1-m)*f_base + m*f_source
                        out_heads[:, :seq_len, head_idx, :] = (
                            (1 - m) * base_heads[:, :seq_len, head_idx, :] +
                            m * src[:, :seq_len, :].detach()
                        )

                return (out_heads.view(bs, seq, hidden),) + args[1:]
            return hook_fn

        for layer_idx in range(self.n_layers):
            layer = self.layers_attr()[layer_idx]
            attn = self._get_attention_module(layer)
            proj = attn.o_proj if hasattr(attn, 'o_proj') else attn.c_proj
            hooks.append(proj.register_forward_pre_hook(create_dbm_hook(layer_idx)))

        outputs = self.model(base_ids)
        logits = outputs.logits

        for hook in hooks:
            hook.remove()

        return logits

    def train_circuit_mask(
            self,
            examples: List[Dict],
            n_epochs: int = 20,
            lr: float = 0.001,
            batch_size: int = 16,
            temp_start: float = 10.0,
            temp_end: float = 0.1,
            lambda_sparsity: float = 0.1,
    ) -> List[CircuitScore]:
        """
        Train a DBM to identify the task circuit (task-general, not hypothesis-specific).

        Same DBM algorithm as train_dcm_mask but uses task counterfactual examples
        directly rather than hypothesis triplets. Returns List[CircuitScore] of
        active heads (mask > 0.5 at T_final), sorted by mask value descending.
        The score field holds the final mask value in (0, 1).

        lambda_sparsity: L1 penalty on mask weights to push masks toward 0 (sparse circuit).
        """
        print(f"\n{'='*60}")
        print(f"TRAINING TASK CIRCUIT MASK (DBM)")
        print(f"  Epochs={n_epochs}, lr={lr}, batch={batch_size}, T: {temp_start}→{temp_end}, lambda_sparsity={lambda_sparsity}")
        print(f"{'='*60}")

        # Convert counterfactual examples → triplets
        triplets = []
        for ex in examples:
            q   = ex.get('question', '')
            cf  = ex.get('counterfactual_question', '')
            ans = str(ex.get('answer', ''))
            if q and cf and ans:
                triplets.append({
                    'original':       q,
                    'counterfactual': cf,
                    'target':         ans,
                    'hypothesis':     'task_circuit',
                })

        if not triplets:
            print("  ⚠️ No valid triplets — returning empty circuit")
            return []

        print(f"  Using {len(triplets)} triplets")

        import random as _random
        n_total_heads = self.n_layers * self.n_heads
        mask_logits = torch.zeros(n_total_heads, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([mask_logits], lr=lr)

        best_loss = float('inf')
        best_mask_logits = mask_logits.detach().clone()

        for epoch in range(n_epochs):
            T = temp_start + (temp_end - temp_start) * epoch / max(n_epochs - 1, 1)
            mask = torch.sigmoid(mask_logits / T)

            batch = _random.sample(triplets, min(batch_size, len(triplets)))
            epoch_loss = torch.tensor(0.0, device=self.device)
            valid = 0

            for triplet in batch:
                orig_ids = self.tokenizer(
                    triplet['original'], return_tensors="pt",
                    truncation=True, max_length=512
                ).input_ids.to(self.device)
                cf_ids = self.tokenizer(
                    triplet['counterfactual'], return_tensors="pt",
                    truncation=True, max_length=512
                ).input_ids.to(self.device)
                target_ids = self.tokenizer(
                    triplet['target'], return_tensors="pt",
                    add_special_tokens=False
                ).input_ids.to(self.device)

                if target_ids.shape[1] == 0:
                    continue

                target_token = target_ids[0, 0].item()
                logits = self._forward_with_dbm(orig_ids, cf_ids, mask)
                log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                ce_loss = -log_probs[target_token]
                sparsity_loss = lambda_sparsity * mask.sum()
                epoch_loss = epoch_loss + ce_loss + sparsity_loss
                valid += 1

            if valid == 0:
                continue

            avg_loss = epoch_loss / valid
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_([mask_logits], max_norm=1.0)
            optimizer.step()

            loss_val = avg_loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_mask_logits = mask_logits.detach().clone()

            active = (torch.sigmoid(mask_logits / T) > 0.5).sum().item()
            print(f"  Epoch {epoch+1:2d}/{n_epochs}: T={T:.2f}  loss={loss_val:.4f}  active={active}")

        final_mask = torch.sigmoid(best_mask_logits / temp_end)

        scores = [
            CircuitScore(
                layer=layer_idx,
                head=head_idx,
                score=final_mask[layer_idx * self.n_heads + head_idx].item()
            )
            for layer_idx in range(self.n_layers)
            for head_idx in range(self.n_heads)
        ]

        active_scores = sorted(
            [s for s in scores if s.score > 0.5],
            key=lambda x: x.score, reverse=True
        )

        print(f"\nTask circuit: {len(active_scores)}/{n_total_heads} active heads (mask > 0.5 at T={temp_end})")
        print(f"Best loss: {best_loss:.4f}")
        if active_scores:
            print(f"Top heads: {[(s.layer, s.head, round(s.score, 3)) for s in active_scores[:5]]}")

        return active_scores

    @torch.no_grad()
    def _eval_with_fixed_mask(
            self,
            examples: List[Dict],
            binary_mask: torch.Tensor,
            max_examples: int = 30,
    ) -> float:
        """
        Evaluate mean log-prob using a fixed binary DBM mask.

        binary_mask: shape (n_layers * n_heads,), values 0.0 or 1.0
          0 → head uses base activations (kept / normal)
          1 → head uses source activations (disrupted / off)

        Source for each example is its paired counterfactual question.
        Returns mean log-prob across examples.
        """
        log_probs = []
        for ex in examples[:max_examples]:
            question  = ex['question']
            cf        = ex.get('counterfactual_question', question)
            answer    = str(ex['answer']).strip()

            full_ids = self.tokenizer(
                f"{question} {answer}", return_tensors="pt",
                truncation=True, max_length=512
            ).input_ids.to(self.device)
            cf_ids = self.tokenizer(
                f"{cf} {answer}", return_tensors="pt",
                truncation=True, max_length=512
            ).input_ids.to(self.device)
            q_ids = self.tokenizer(
                question, return_tensors="pt",
                truncation=True, max_length=512
            ).input_ids
            ans_ids = self.tokenizer(
                answer, return_tensors="pt", add_special_tokens=False
            ).input_ids
            if ans_ids.shape[1] == 0:
                continue

            logits = self._forward_with_dbm(full_ids, cf_ids, binary_mask)
            lp = 0.0
            valid = 0
            target_tokens = ans_ids[0].tolist()
            ans_start = q_ids.shape[1]
            for i, tok in enumerate(target_tokens):
                pos = ans_start + i - 1
                if 0 <= pos < logits.shape[1]:
                    lp += torch.log_softmax(logits[0, pos, :], dim=-1)[tok].item()
                    valid += 1
            if valid > 0:
                log_probs.append(lp / valid)

        return float(np.mean(log_probs)) if log_probs else 0.0

    def compute_faithfulness_dbm(
            self,
            circuit: List[CircuitScore],
            examples: List[Dict],
            max_examples: int = 30,
    ) -> Dict:
        """
        Mask-based faithfulness (sufficiency of the circuit).

        Full model:    all mask=0 (all heads use base activations)
        Circuit only:  circuit heads mask=0, non-circuit heads mask=1 (disrupted)

        Faithfulness = mean_logprob(circuit only) / mean_logprob(full model)
        """
        print(f"\n{'='*60}")
        print(f"FAITHFULNESS (mask-based DBM)")
        print(f"{'='*60}")

        n_total = self.n_layers * self.n_heads
        circuit_set = {(s.layer, s.head) for s in circuit}

        # Full model: no disruption
        full_mask = torch.zeros(n_total, device=self.device)
        f_m = self._eval_with_fixed_mask(examples, full_mask, max_examples)

        # Circuit only: disrupt all non-circuit heads
        circuit_mask = torch.ones(n_total, device=self.device)
        for layer_idx, head_idx in circuit_set:
            circuit_mask[layer_idx * self.n_heads + head_idx] = 0.0
        f_c_m = self._eval_with_fixed_mask(examples, circuit_mask, max_examples)

        faithfulness = f_c_m / (f_m + 1e-10) if f_m != 0 else 0.0

        print(f"  Full model log-prob:     {f_m:.4f}")
        print(f"  Circuit-only log-prob:   {f_c_m:.4f}")
        print(f"  Faithfulness:            {faithfulness:.4f}")
        print(f"  Circuit size:            {len(circuit_set)} / {n_total} heads")

        return {
            'faithfulness': faithfulness,
            'f_m': f_m,
            'f_c_m': f_c_m,
            'circuit_size': len(circuit_set),
            'total_heads': n_total,
            'circuit_fraction': len(circuit_set) / n_total,
            'examples_evaluated': min(len(examples), max_examples),
        }

    def compute_necessity_sufficiency_dbm(
            self,
            circuit: List[CircuitScore],
            examples: List[Dict],
            max_examples: int = 30,
    ) -> Dict:
        """
        Mask-based necessity and sufficiency — no ablation, no path patching.

        Uses _forward_with_dbm with fixed binary masks:

          Necessity of head h   = logprob(full) - logprob(h disrupted, others kept)
                                  High drop → h is necessary.

          Sufficiency of head h = logprob(only h kept, all others disrupted)
                                  High → h alone is sufficient.

        Circuit-level:
          Circuit necessity   = logprob(full) - logprob(circuit disrupted, others kept)
          Circuit sufficiency = faithfulness  = logprob(circuit kept) / logprob(full)
        """
        print(f"\n{'='*60}")
        print(f"NECESSITY & SUFFICIENCY (mask-based DBM)")
        print(f"{'='*60}")

        n_total = self.n_layers * self.n_heads
        circuit_heads = [(s.layer, s.head) for s in circuit]
        circuit_set   = set(circuit_heads)

        if not circuit_heads:
            print("  No circuit heads — skipping")
            return {}

        # Full model baseline (all mask=0)
        full_mask = torch.zeros(n_total, device=self.device)
        print("  Full model baseline...")
        full_mean = self._eval_with_fixed_mask(examples, full_mask, max_examples)

        # Circuit sufficiency (circuit kept, non-circuit disrupted)
        circuit_only_mask = torch.ones(n_total, device=self.device)
        for l, h in circuit_set:
            circuit_only_mask[l * self.n_heads + h] = 0.0
        print("  Circuit sufficiency...")
        circuit_only_mean = self._eval_with_fixed_mask(examples, circuit_only_mask, max_examples)

        # Circuit necessity (circuit disrupted, non-circuit kept)
        no_circuit_mask = torch.zeros(n_total, device=self.device)
        for l, h in circuit_set:
            no_circuit_mask[l * self.n_heads + h] = 1.0
        print("  Circuit necessity...")
        no_circuit_mean = self._eval_with_fixed_mask(examples, no_circuit_mask, max_examples)

        circuit_necessity   = full_mean - no_circuit_mean
        circuit_sufficiency = circuit_only_mean / (full_mean + 1e-10) if full_mean != 0 else 0.0

        print(f"  Full model:       {full_mean:.4f}")
        print(f"  Circuit only:     {circuit_only_mean:.4f}  (sufficiency)")
        print(f"  No circuit:       {no_circuit_mean:.4f}  (necessity)")
        print(f"  Circuit necessity drop: {circuit_necessity:.4f}")
        print(f"  Circuit sufficiency:    {circuit_sufficiency:.4f}")

        # Per-head scores
        per_head = {}
        for i, (layer_idx, head_idx) in enumerate(circuit_heads):
            print(f"  Head {i+1}/{len(circuit_heads)}: L{layer_idx}H{head_idx}")

            # Necessity: disrupt only this head
            nec_mask = torch.zeros(n_total, device=self.device)
            nec_mask[layer_idx * self.n_heads + head_idx] = 1.0
            minus_one_mean = self._eval_with_fixed_mask(examples, nec_mask, max_examples)
            necessity = full_mean - minus_one_mean

            # Sufficiency: keep only this head, disrupt all others
            suf_mask = torch.ones(n_total, device=self.device)
            suf_mask[layer_idx * self.n_heads + head_idx] = 0.0
            solo_mean = self._eval_with_fixed_mask(examples, suf_mask, max_examples)

            per_head[f"L{layer_idx}H{head_idx}"] = {
                'layer': layer_idx,
                'head': head_idx,
                'mask_value': float(circuit[i].score),
                'necessity': float(necessity),
                'sufficiency_logprob': float(solo_mean),
                'full_model_logprob': float(full_mean),
            }
            print(f"    necessity={necessity:.4f}  sufficiency_lp={solo_mean:.4f}")

        return {
            'circuit_necessity':   float(circuit_necessity),
            'circuit_sufficiency': float(circuit_sufficiency),
            'full_model_logprob':  float(full_mean),
            'circuit_only_logprob': float(circuit_only_mean),
            'no_circuit_logprob':  float(no_circuit_mean),
            'circuit_size':        len(circuit_heads),
            'per_head':            per_head,
        }

    def compare_circuits_dbm(
            self,
            circuits: Dict[str, List[CircuitScore]],
    ) -> Dict:
        """
        Mask-based cross-model circuit comparison — replaces CMAP.

        For every head in the base circuit, compares its mask value across
        models. A head with high mask value in RL but low in SFT is better
        preserved by RL.

        Args:
            circuits: {'base': [...], 'sft': [...], 'rl': [...]} — mask values
                      come from train_circuit_mask (score = final DBM mask value)
        """
        base_circuit = circuits.get('base', [])
        if not base_circuit:
            return {}

        # Build head → mask_value lookup for each model
        def mask_lookup(circuit):
            return {(s.layer, s.head): s.score for s in circuit}

        lookups = {label: mask_lookup(c) for label, c in circuits.items()}
        base_lookup = lookups.get('base', {})

        head_comparison = []
        for s in base_circuit:
            key = (s.layer, s.head)
            row = {
                'layer': s.layer,
                'head': s.head,
                'base_mask': float(s.score),
            }
            for label, lkp in lookups.items():
                if label == 'base':
                    continue
                row[f'{label}_mask'] = float(lkp.get(key, 0.0))
                row[f'{label}_delta'] = float(lkp.get(key, 0.0)) - float(s.score)
            head_comparison.append(row)

        # Which heads are better preserved by each fine-tuned model?
        model_labels = [l for l in circuits if l != 'base']
        vulnerable = {}
        for label in model_labels:
            other_labels = [l for l in model_labels if l != label]
            for other in other_labels:
                key = f"{label}_better_than_{other}"
                vulnerable[key] = [
                    h for h in head_comparison
                    if h.get(f'{label}_mask', 0) > h.get(f'{other}_mask', 0)
                ]

        return {
            'head_comparison': head_comparison,
            'vulnerable': vulnerable,
        }

    def analyze_all_hypotheses(
            self,
            dataset,
            n_examples: int = 500,
            dataset_type: str = 'science'
    ) -> Dict[str, DCMResult]:
        """Run DCM analysis for all functionality hypotheses.

        Args:
            dataset: The HuggingFace dataset (items must match the task format)
            n_examples: Number of triplets per hypothesis (paper uses 500)
            dataset_type: 'science' uses chemistry hypotheses
                          (answer_key, molecule, task_type);
                          'math' uses arithmetic hypotheses
                          (position, value, operation)
        """
        if dataset_type == 'science':
            hypotheses = ["answer_key", "molecule", "task_type"]
            create_triplets = self.create_dcm_triplets_science
        else:
            hypotheses = ["position", "value", "operation"]
            create_triplets = self.create_dcm_triplets_math

        results = {}

        for hypothesis in hypotheses:
            print(f"\n{'='*70}")
            print(f"Analyzing hypothesis: {hypothesis}")
            print(f"{'='*70}")

            triplets = create_triplets(dataset, hypothesis, n_examples)

            if len(triplets) < 5:
                print(f"  ⚠️ Not enough triplets for {hypothesis}, skipping")
                continue

            result = self.train_dcm_mask(triplets)
            results[hypothesis] = result

        return results


def create_counterfactual_examples_math(dataset, n_examples: int = 100) -> List[Dict]:
    """
    Create meaningful counterfactuals for math problems by changing numbers.

    FIXED: Uses value increments instead of position swaps for commutative operations,
    ensuring the counterfactual actually changes the expected answer.
    """
    import random

    examples = []

    for i in range(min(n_examples * 3, len(dataset))):  # Sample more to filter
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
            # FIXED: Use value increment as primary strategy
            # This changes the answer for ALL operations (not just non-commutative)
            num = numbers[0]
            try:
                if '.' in num:
                    new_num = str(float(num) + 1)
                else:
                    new_num = str(int(num) + 1)

                counterfactual = question.replace(num, new_num, 1)

                if counterfactual != question:
                    examples.append({
                        'question': question,
                        'answer': str(answer),
                        'counterfactual_question': counterfactual,
                        'counterfactual_type': 'value_increment',
                        'changed_value': (num, new_num)
                    })
            except ValueError:
                pass

        # Secondary strategy: For non-commutative operations, also try position swap
        if len(numbers) >= 2:
            # Check if operation is non-commutative (subtraction, division)
            has_noncommutative = any(op in question.lower() for op in ['-', '/', 'minus', 'subtract', 'divided', 'from'])

            if has_noncommutative:
                num1, num2 = numbers[0], numbers[1]
                # Position swap for non-commutative operations
                counterfactual = question.replace(num1, "<<TEMP>>")
                counterfactual = counterfactual.replace(num2, num1)
                counterfactual = counterfactual.replace("<<TEMP>>", num2)

                if counterfactual != question:
                    examples.append({
                        'question': question,
                        'answer': str(answer),
                        'counterfactual_question': counterfactual,
                        'counterfactual_type': 'position_swap',
                        'original_numbers': numbers
                    })

        if len(examples) >= n_examples:
            break

    print(f"\n✅ Created {len(examples)} counterfactual pairs")
    if examples:
        # Count by type
        by_type = {}
        for ex in examples:
            t = ex.get('counterfactual_type', 'unknown')
            by_type[t] = by_type.get(t, 0) + 1
        print(f"   By type: {by_type}")

        print("\n📋 Sample counterfactuals:")
        for i, ex in enumerate(examples[:3]):
            print(f"  {i+1}. Type: {ex['counterfactual_type']}")
            print(f"     Original: {ex['question']}")
            print(f"     Counterfactual: {ex['counterfactual_question']}")
            print(f"     Answer: {ex['answer']}")

    return examples


def create_counterfactual_examples_science(dataset, n_examples: int = 100) -> List[Dict]:
    """
    Create counterfactual examples for SciKnowEval chemistry dataset.
    Uses answer_key rotation: rotate MCQ choices left by 1 so the correct
    answer shifts to a different letter.

    Returns list of dicts with keys: question, answer, counterfactual_question
    (same format as create_counterfactual_examples_math, for circuit discovery).
    """
    labels_order = ["A", "B", "C", "D"]
    examples = []

    for item in dataset:
        if item.get('type') != 'mcq-4-choices':
            continue
        texts = item.get('choices', {}).get('text', [])
        answer_key = item.get('answerKey', '')
        question = item.get('question', '')

        if answer_key not in labels_order or len(texts) != 4 or not question:
            continue

        correct_idx = labels_order.index(answer_key)
        rotated_texts = texts[1:] + texts[:1]
        new_correct_idx = (correct_idx - 1) % 4
        new_correct_key = labels_order[new_correct_idx]

        orig_opts = "\n".join(f"{l}: {t}" for l, t in zip(labels_order, texts))
        cf_opts   = "\n".join(f"{l}: {t}" for l, t in zip(labels_order, rotated_texts))

        examples.append({
            'question':               f"{question}\n{orig_opts}",
            'answer':                 answer_key,
            'counterfactual_question': f"{question}\n{cf_opts}",
        })

        if len(examples) >= n_examples:
            break

    print(f"Created {len(examples)} science counterfactual examples")
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