import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name, device="auto"):
    """
    Load model and tokenizer with appropriate settings.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device mapping strategy
        
    Returns:
        tuple: (model, tokenizer)
    """
    print("\n" + "="*70)
    print("LOADING MODEL AND TOKENIZER")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Device mapping: {device}")
    print("\n→ Loading model from HuggingFace (this may take a while)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16
        device_map=device,
        trust_remote_code=True,
    )
    
    print(" Model loaded successfully")
    print("\n Loading tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(" Tokenizer loaded successfully")
    
    print(f"\n Model Statistics:")
    print(f" Total parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f" Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")
    
    print("\n" + "="*70)
    print("MODEL LOADING COMPLETE")
    print("="*70)
    
    return model, tokenizer


def check_device():
    """
    Check and print GPU/device information.
    """
    print("\n" + "="*70)
    print("CHECKING DEVICE AVAILABILITY")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    
    if torch.cuda.is_available():
        print(f" GPU is available!")
        print(f" GPU Name: {torch.cuda.get_device_name(0)}")
        print(f" Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f" Current Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f" Current Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    else:
        print(" No GPU available, using CPU (training will be very slow)")
    
    print("="*70)
    
    return device