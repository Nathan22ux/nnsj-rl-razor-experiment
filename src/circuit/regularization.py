"""
Circuit-Aware Regularization for SFT (Section 2.5 of paper).
Uses identified vulnerable circuits to regularize SFT training.

This implements Equation 6:
L_total = L_SFT + λ * Σ ||a^π_h - a^π0_h||²

Where the sum is over vulnerable circuit heads identified by CMAP analysis.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass


@dataclass
class VulnerableHead:
    """Represents a vulnerable circuit head"""
    layer: int
    head: int
    vulnerability: float


class CircuitAwareRegularizer(nn.Module):
    """
    Regularizer that penalizes changes to vulnerable circuit heads.
    
    This helps SFT preserve important circuits that RL naturally preserves
    due to KL divergence minimization.
    """
    
    def __init__(
        self,
        base_model,
        vulnerable_heads: List[VulnerableHead],
        lambda_reg: float = 0.01,
        device: str = "cuda"
    ):
        super().__init__()
        self.base_model = base_model
        self.vulnerable_heads = vulnerable_heads
        self.lambda_reg = lambda_reg
        self.device = device
        
        # Cache base model activations (we'll compute them once per batch)
        self.base_activations = {}
        
        print(f"\nInitialized CircuitAwareRegularizer:")
        print(f"  Vulnerable heads: {len(vulnerable_heads)}")
        print(f"  Regularization strength (λ): {lambda_reg}")
        
        # Print vulnerable heads
        print("\n  Regularizing heads:")
        for i, head in enumerate(vulnerable_heads[:5], 1):
            print(f"    {i}. Layer {head.layer}, Head {head.head} (vulnerability: {head.vulnerability:.4f})")
        if len(vulnerable_heads) > 5:
            print(f"    ... and {len(vulnerable_heads) - 5} more")
    
    def extract_head_activation(self, model, input_ids, layer_idx: int, head_idx: int):
        """
        Extract activation for a specific attention head.
        """
        activation = None
        
        def hook_fn(module, input, output):
            nonlocal activation
            attn_output = output[0]
            batch_size, seq_len, hidden_dim = attn_output.shape
            n_heads = model.config.num_attention_heads
            head_dim = hidden_dim // n_heads
            
            # Reshape to separate heads
            attn_output_heads = attn_output.reshape(batch_size, seq_len, n_heads, head_dim)
            
            # Extract this specific head
            activation = attn_output_heads[:, :, head_idx, :].clone()
        
        # Register hook
        layer = model.model.layers[layer_idx]
        hook = layer.self_attn.o_proj.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            model(input_ids)
        
        # Remove hook
        hook.remove()
        
        return activation
    
    def compute_regularization_loss(self, fine_tuned_model, input_ids):
        """
        Compute regularization loss for vulnerable heads.
        
        L_reg = Σ ||a^π_h - a^π0_h||²
        """
        total_loss = 0.0
        
        for vulnerable_head in self.vulnerable_heads:
            layer_idx = vulnerable_head.layer
            head_idx = vulnerable_head.head
            
            # Get base model activation
            base_activation = self.extract_head_activation(
                self.base_model,
                input_ids,
                layer_idx,
                head_idx
            )
            
            # Get fine-tuned model activation
            ft_activation = self.extract_head_activation(
                fine_tuned_model,
                input_ids,
                layer_idx,
                head_idx
            )
            
            # Compute L2 distance
            head_loss = torch.norm(ft_activation - base_activation, p=2) ** 2
            
            # Weight by vulnerability score (optional)
            # head_loss *= vulnerable_head.vulnerability
            
            total_loss += head_loss
        
        return total_loss


class CircuitAwareTrainer(Trainer):
    """
    Custom Trainer that adds circuit-aware regularization to SFT loss.
    
    Implements: L_total = L_SFT + λ * L_reg
    """
    
    def __init__(
        self,
        regularizer: CircuitAwareRegularizer,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.regularizer = regularizer
        self.reg_losses = []  # Track regularization losses for logging
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute total loss with circuit regularization.
        """
        # Standard SFT loss
        outputs = model(**inputs)
        sft_loss = outputs.loss
        
        # Circuit regularization loss
        input_ids = inputs.get('input_ids')
        if input_ids is not None and self.regularizer.lambda_reg > 0:
            reg_loss = self.regularizer.compute_regularization_loss(model, input_ids)
            reg_loss = self.regularizer.lambda_reg * reg_loss
            
            # Track for logging
            self.reg_losses.append(reg_loss.item())
        else:
            reg_loss = 0.0
        
        # Total loss
        total_loss = sft_loss + reg_loss
        
        # Log both losses
        if self.state.global_step % 10 == 0:
            print(f"\nStep {self.state.global_step}: SFT Loss = {sft_loss.item():.4f}, "
                  f"Reg Loss = {reg_loss if isinstance(reg_loss, float) else reg_loss.item():.4f}, "
                  f"Total = {total_loss.item():.4f}")
        
        return (total_loss, outputs) if return_outputs else total_loss


def train_sft_with_circuit_regularization(
    model,
    base_model,
    dataset,
    tokenizer,
    vulnerable_heads: List[VulnerableHead],
    lambda_reg: float = 0.01,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    epochs: int = 3,
    output_dir: str = "models/sft_circuit_aware"
):
    """
    Train SFT model with circuit-aware regularization.
    
    Args:
        model: Model to fine-tune
        base_model: Base model for computing regularization
        dataset: Training dataset
        tokenizer: Tokenizer
        vulnerable_heads: List of vulnerable heads to regularize
        lambda_reg: Regularization strength
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Number of epochs
        output_dir: Output directory for checkpoints
    
    Returns:
        Trained model and trainer
    """
    print("\n" + "="*70)
    print("TRAINING SFT WITH CIRCUIT-AWARE REGULARIZATION")
    print("="*70)
    
    # Initialize regularizer
    regularizer = CircuitAwareRegularizer(
        base_model=base_model,
        vulnerable_heads=vulnerable_heads,
        lambda_reg=lambda_reg
    )
    
    # Prepare dataset
    def formatting_func(examples):
        texts = []
        for i in range(len(examples['0'])):
            question = examples['0'][i]['value']
            try:
                answer = examples['1'][i]['ground_truth']['value']
            except (KeyError, TypeError):
                answer = str(examples['1'][i])
            text = f"Question: {question}\nAnswer: {answer}"
            texts.append(text)
        return {'text': texts}
    
    formatted_dataset = dataset.map(
        formatting_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        bf16=True,
        remove_unused_columns=False,
    )
    
    # Create trainer with circuit regularization
    trainer = CircuitAwareTrainer(
        regularizer=regularizer,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Print average regularization loss
    if trainer.reg_losses:
        avg_reg_loss = sum(trainer.reg_losses) / len(trainer.reg_losses)
        print(f"\nAverage regularization loss: {avg_reg_loss:.4f}")
    
    return model, trainer


def load_vulnerable_heads_from_analysis(results_path: str) -> List[VulnerableHead]:
    """
    Load vulnerable heads from circuit analysis results.
    
    Args:
        results_path: Path to circuit analysis JSON file
    
    Returns:
        List of VulnerableHead objects
    """
    import json
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    vulnerable_heads = []
    for head_info in results['vulnerable_circuits']:
        vulnerable_heads.append(VulnerableHead(
            layer=head_info['layer'],
            head=head_info['head'],
            vulnerability=head_info['vulnerability']
        ))
    
    print(f"\nLoaded {len(vulnerable_heads)} vulnerable heads from {results_path}")
    
    return vulnerable_heads


# Example usage
if __name__ == "__main__":
    """
    Example of how to use circuit-aware regularization.
    """
    print("""
    Example Usage:
    
    1. First, run circuit analysis to identify vulnerable heads:
       python run_circuit_analysis.py --task math --sft_checkpoint ... --rl_checkpoint ...
    
    2. Load vulnerable heads and train with regularization:
       
       from circuit_regularization import (
           load_vulnerable_heads_from_analysis,
           train_sft_with_circuit_regularization
       )
       
       # Load vulnerable heads
       vulnerable_heads = load_vulnerable_heads_from_analysis(
           'results/circuits/circuit_analysis_math.json'
       )
       
       # Load models
       base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
       sft_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
       
       # Train with regularization
       trained_model, trainer = train_sft_with_circuit_regularization(
           model=sft_model,
           base_model=base_model,
           dataset=dataset,
           tokenizer=tokenizer,
           vulnerable_heads=vulnerable_heads,
           lambda_reg=0.01,  # Tune this parameter
       )
    
    3. Evaluate and compare:
       - Standard SFT
       - RL
       - SFT + Circuit Regularization
    """)