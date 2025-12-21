"""
Modified main.py for RL's Razor Replication

ENTRY POINT for fixed implementation

Usage:
    python main.py                    # Run with default config
    python main.py --mode quick       # Quick test
    python main.py --mode minimal     # Budget replication
    python main.py --mode full        # Full paper replication

This integrates all fixed components:
- evaluation.py
- training.py
- config.py
- dataset_utils.py
- modified_experiment.py
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# Unbuffered output for real-time logging
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# Allow HuggingFace code evaluation
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

print("="*80)
print(" RL'S RAZOR REPLICATION - FIXED IMPLEMENTATION")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

print(" Importing modules...")

try:
    from config import (
        MODEL_NAME, 
        get_config, 
        print_config_summary,
        count_total_runs,
        estimate_compute_hours
    )
    from dataset_utils import (
        load_open_reasoner_zero,
        load_alpaca,
        load_science_dataset
    )
    from experiment import run_full_experiment
    
    # Import transformers components
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(" All modules imported successfully\n")

except ImportError as e:
    print(f" Import error: {e}")
    print("\nMake sure you have:")
    print("1. Replaced src/evaluation.py with evaluation.py")
    print("2. Replaced src/training.py with training.py")
    print("3. Replaced src/config.py with config.py")
    print("4. Added src/dataset_utils.py")
    print("5. Added this modified_experiment.py")
    print("\nRun: python tests/test_all_fixes.py to verify installation\n")
    sys.exit(1)


def check_gpu():
    """Check GPU availability and memory"""
    print("="*80)
    print(" GPU CHECK")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f" CUDA available")
        print(f"   Device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3  # GB
            print(f"   GPU {i}: {props.name}")
            print(f"    Memory: {total_memory:.1f} GB")
        
        # Check current memory usage
        current_memory = torch.cuda.memory_allocated(0) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"\n   Current memory usage: {current_memory:.2f} GB")
        print(f"   Max memory usage: {max_memory:.2f} GB")
        
        return True
    else:
        print(" No CUDA devices available")
        print("  This will run on CPU (very slow)")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with error handling"""
    print(f"\n{'='*80}")
    print(f" LOADING MODEL AND TOKENIZER")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    
    try:
        print("\n Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   Set pad_token = eos_token")
        
        print(" Tokenizer loaded")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Model max length: {tokenizer.model_max_length}")
        
        print("\n Loading model...")
        print("  This may take a few minutes...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        print(" Model loaded")
        print(f"   Parameters: {model.num_parameters() / 1e9:.2f}B")
        print(f"   Device: {model.device}")
        print(f"   Dtype: {model.dtype}")
        
        # Test model
        print("\n Testing model...")
        test_input = tokenizer("Hello world", return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**test_input, max_new_tokens=5)
        test_output = tokenizer.decode(output[0])
        print(f"  Test generation: {test_output[:50]}...")
        print(" Model working correctly")
        
        print(f"{'='*80}\n")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"\n Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check you have internet connection (for downloading)")
        print("2. Check you have sufficient GPU memory (need >40GB for 3B model)")
        print("3. Try a smaller model like 'gpt2'")
        print("4. Check HuggingFace authentication if model is gated\n")
        raise


def load_dataset(dataset_name, config_mode):
    """Load and prepare dataset"""
    print(f"\n{'='*80}")
    print(f" LOADING DATASET")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    
    try:
        if dataset_name == "math":
            print("\n Attempting to load Open-Reasoner-Zero...")
            print("  (Will fall back to GSM8K if unavailable)")
            dataset = load_open_reasoner_zero()
        
        elif dataset_name == "alpaca":
            print("\n Loading Alpaca dataset...")
            dataset = load_alpaca()
        
        elif dataset_name == "science":
            print("\n Attempting to load SciKnowEval...")
            print("  (Will fall back to SciQ if unavailable)")
            dataset = load_science_dataset()
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"\n Dataset loaded successfully")
        print(f"   Total examples: {len(dataset)}")
        print(f"   Columns: {dataset.column_names}")
        
        # Show example
        print(f"\n Example from dataset:")
        example = dataset[0]
        if 'question' in example:
            print(f"  Question: {example['question'][:100]}...")
            print(f"  Answer: {example['answer'][:100]}...")
        elif 'text' in example:
            print(f"  Text: {example['text'][:200]}...")
        
        print(f"{'='*80}\n")
        
        return dataset
    
    except Exception as e:
        print(f"\n Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try a different dataset: 'math', 'alpaca', or 'science'")
        print("3. Check HuggingFace datasets library is installed: pip install datasets\n")
        raise


def create_visualizations(results, dataset_name):
    """Create Pareto frontier plot"""
    print(f"\n{'='*80}")
    print(f" CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        sns.set_style("whitegrid")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Pareto Frontier (NT vs PT)
        if results['sft']:
            sft_nt = [r['NT'] for r in results['sft']]
            sft_pt = [r['PT'] for r in results['sft']]
            ax1.scatter(sft_nt, sft_pt, alpha=0.6, s=100, label='SFT', color='red')
        
        if results['rl']:
            rl_nt = [r['NT'] for r in results['rl']]
            rl_pt = [r['PT'] for r in results['rl']]
            ax1.scatter(rl_nt, rl_pt, alpha=0.6, s=100, label='RL', color='blue')
        
        ax1.set_xlabel('New Task Accuracy (%)', fontsize=12)
        ax1.set_ylabel('Prior Task Performance', fontsize=12)
        ax1.set_title('Pareto Frontier: Forgetting vs Learning', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: KL Divergence
        if results['sft']:
            sft_kl = [r['kl_divergence'] for r in results['sft']]
            ax2.scatter(sft_nt, sft_kl, alpha=0.6, s=100, label='SFT', color='red')
        
        if results['rl']:
            rl_kl = [r['kl_divergence'] for r in results['rl']]
            ax2.scatter(rl_nt, rl_kl, alpha=0.6, s=100, label='RL', color='blue')
        
        ax2.set_xlabel('New Task Accuracy (%)', fontsize=12)
        ax2.set_ylabel('KL Divergence from Base Model', fontsize=12)
        ax2.set_title('Distributional Shift', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs('plots', exist_ok=True)
        plot_file = f'plots/pareto_frontier_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        print(f" Visualization saved to: {plot_file}")
        print(f"{'='*80}\n")
        
    except ImportError:
        print(" matplotlib/seaborn not installed, skipping visualization")
        print("  Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f" Error creating visualization: {e}")


def main():
    """Main entry point"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="RL's Razor Replication (Fixed)")
    parser.add_argument(
        '--mode',
        type=str,
        default='minimal',
        choices=['quick', 'minimal', 'full'],
        help='Configuration mode (default: minimal)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='math',
        choices=['math', 'alpaca', 'science'],
        help='Dataset to use (default: math)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model to use (default: from config)'
    )
    parser.add_argument(
        '--skip-gpu-check',
        action='store_true',
        help='Skip GPU availability check'
    )
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"   Mode: {args.mode}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.model or MODEL_NAME}")
    print()
    
    # Show configuration summary
    print_config_summary(args.mode)
    
    # Estimate compute
    compute = estimate_compute_hours(args.mode, '3B')
    print(f"Estimated compute requirements:")
    print(f"   Total GPU hours: {compute['total_hours']}")
    print(f"   Estimated cost: ${compute['estimated_cost_usd']}")
    print()
    
    # Confirm
    response = input("Proceed with experiment? (y/n): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    # GPU check
    if not args.skip_gpu_check:
        if not check_gpu():
            print("GPU check failed. Exiting.")
            return
    
    # Use custom model if specified
    model_name = args.model or MODEL_NAME
    
    # Load model and tokenizer (load once, use for all experiments)
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Load dataset
    dataset = load_dataset(args.dataset, args.mode)
    
    # Clean up initial model (will be loaded fresh in experiment)
    del model
    torch.cuda.empty_cache()
    
    # Run experiment
    print(f"\n{'='*80}")
    print(f" STARTING EXPERIMENT")
    print(f"{'='*80}\n")
    
    results = run_full_experiment(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        config_mode=args.mode
    )
    
    # Create visualizations
    create_visualizations(results, args.dataset)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f" EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results['sft'] and results['rl']:
        import numpy as np
        
        sft_pt = np.mean([r['PT'] for r in results['sft']])
        rl_pt = np.mean([r['PT'] for r in results['rl']])
        sft_kl = np.mean([r['kl_divergence'] for r in results['sft']])
        rl_kl = np.mean([r['kl_divergence'] for r in results['rl']])
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"\nPrior Task Performance (higher = better retention):")
        print(f"   SFT: {sft_pt:.4f}")
        print(f"   RL:  {rl_pt:.4f}")
        print(f"   Improvement: {(rl_pt - sft_pt) / sft_pt * 100:+.2f}%")
        
        print(f"\nKL Divergence (lower = less forgetting):")
        print(f"   SFT: {sft_kl:.4f}")
        print(f"   RL:  {rl_kl:.4f}")
        print(f"   Reduction: {(sft_kl - rl_kl) / sft_kl * 100:.2f}%")
        
        # Validate hypothesis
        hypothesis_validated = (rl_pt > sft_pt) and (rl_kl < sft_kl)
        
        print(f"\n{'='*80}")
        if hypothesis_validated:
            print(f"  HYPOTHESIS VALIDATED ")
            print(f" RL preserves prior knowledge better than SFT")
            print(f" (Higher PT and Lower KL)")
        else:
            print(f"  UNEXPECTED RESULTS")
            if rl_pt <= sft_pt:
                print(f" RL prior task performance not better than SFT")
            if rl_kl >= sft_kl:
                print(f" RL KL divergence not lower than SFT")
            print(f" Check for implementation issues or data quality")
        print(f"{'='*80}")
    
    print(f"\n All results saved to results/ directory")
    print(f" Visualizations saved to plots/ directory")
    print(f"\nThank you for using the fixed implementation! 🚀\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print("Partial results have been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)