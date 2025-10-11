from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


DEFAULT_MODEL_NAME = "gpt2"


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    torch_dtype: torch.dtype | None = None,
    device_map: str | dict[str, int] | None = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a causal LM and tokenizer, configuring padding and special tokens.

    Args:
        model_name: Hugging Face Hub model ID (e.g., "gpt2" or "EleutherAI/gpt-neo-125M").
        torch_dtype: Optional dtype override for model weights.
        device_map: Device map argument passed to `from_pretrained`.

    Returns:
        Tuple of (model, tokenizer) ready for training or inference.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure the tokenizer has a padding token for batching.
    if tokenizer.pad_token is None:
        # GPT-style tokenizers lack pad/unk by default. Use eos as pad.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    # Align model config with tokenizer updates (pad token etc.).
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer
