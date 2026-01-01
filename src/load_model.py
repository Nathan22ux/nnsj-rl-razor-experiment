"""Module for loading models and tokenizers."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name, device="auto"):
    """
    Load model and tokenizer with appropriate settings.

    Args:
        model_name: HuggingFace model identifier
        device: Device mapping strategy

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info("=" * 70)
    logger.info("LOADING MODEL AND TOKENIZER")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}, Device mapping: {device}")
    logger.info("Loading model from HuggingFace (this may take a while)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16
        device_map=device,
        trust_remote_code=True,
    )

    logger.info("Model loaded successfully")
    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizer loaded successfully")

    total_params = model.num_parameters() / 1e9
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    logger.info(f"Model Statistics: Total parameters: {total_params:.2f}B, "
                f"Trainable parameters: {trainable_params:.2f}B")

    # Optimize torch for speed
    logger.info("Optimizing torch for performance...")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("Torch optimizations applied")

    # Try to compile model for faster inference (may not work with all models)
    try:
        logger.info("Attempting to compile model...")
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model compiled successfully")
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}. Proceeding without compilation.")

    logger.info("=" * 70)
    logger.info("MODEL LOADING COMPLETE")
    logger.info("=" * 70)

    return model, tokenizer


def check_device():
    """Check and return GPU/device information."""
    logger.info("=" * 70)
    logger.info("CHECKING DEVICE AVAILABILITY")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_mem = torch.cuda.memory_allocated() / 1e9
        cached_mem = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU available: {gpu_name}, Total Memory: {total_mem:.2f}GB, "
                    f"Allocated: {allocated_mem:.2f}GB, Cached: {cached_mem:.2f}GB")
    else:
        logger.warning("No GPU available, using CPU (training will be very slow)")

    logger.info("=" * 70)
    return device