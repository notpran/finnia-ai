"""Dataset loading and preprocessing utilities for the mini-GPT project."""
from __future__ import annotations

import os
import random
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer

from config import DataConfig, TrainingConfig

_USER_PREFIX = "User: "
_ASSISTANT_PREFIX = "Assistant: "
_TURN_SUFFIX = "\n"


def _safe_load_dataset(name: str, split: str, cache_dir: Optional[str]) -> Optional[Iterable]:
    try:
        return load_dataset(name, split=split, cache_dir=cache_dir)
    except Exception:
        return None


def _format_oasst(example: dict) -> Optional[str]:
    prompt = example.get("prompt") or example.get("instruction")
    response = example.get("response") or example.get("output")
    if not prompt or not response:
        return None
    return f"{_USER_PREFIX}{prompt}{_TURN_SUFFIX}{_ASSISTANT_PREFIX}{response}{_TURN_SUFFIX}"


def _format_gsm8k(example: dict) -> Optional[str]:
    question = example.get("question")
    answer = example.get("answer")
    if not question or not answer:
        return None
    return f"{_USER_PREFIX}Solve: {question}{_TURN_SUFFIX}{_ASSISTANT_PREFIX}{answer}{_TURN_SUFFIX}"


def _format_code(example: dict) -> Optional[str]:
    prompt = example.get("instruction") or example.get("prompt") or example.get("question")
    completion = example.get("output") or example.get("response") or example.get("code")
    if not prompt or not completion:
        return None
    return f"{_USER_PREFIX}{prompt}{_TURN_SUFFIX}{_ASSISTANT_PREFIX}{completion}{_TURN_SUFFIX}"


def _format_wikitext(example: dict) -> Optional[str]:
    text = example.get("text")
    if not text:
        return None
    return f"{_USER_PREFIX}Paraphrase or continue: {text.strip()}{_TURN_SUFFIX}{_ASSISTANT_PREFIX}"


def _generate_synthetic_math(n: int = 2000) -> List[str]:
    samples = []
    for _ in range(n):
        a, b = random.randint(0, 999), random.randint(0, 999)
        prompt = f"Solve: {a} + {b}"
        solution = f"Step 1: {a} + {b} = {a + b}\nStep 2: Final answer = {a + b}"
        samples.append(f"{_USER_PREFIX}{prompt}{_TURN_SUFFIX}{_ASSISTANT_PREFIX}{solution}{_TURN_SUFFIX}")
    return samples


def _collect_text_samples(cfg: DataConfig) -> List[str]:
    sequences: List[str] = []
    if cfg.dataset_cache_dir:
        os.makedirs(cfg.dataset_cache_dir, exist_ok=True)
    dialog = _safe_load_dataset("OpenAssistant/oasst1", "train", cfg.dataset_cache_dir)
    if dialog:
        for i, row in enumerate(dialog):
            formatted = _format_oasst(row)
            if formatted:
                sequences.append(formatted)
            if cfg.max_samples_per_source and i + 1 >= cfg.max_samples_per_source:
                break

    math_ds = _safe_load_dataset("gsm8k", "train", cfg.dataset_cache_dir)
    if math_ds:
        for i, row in enumerate(math_ds):
            formatted = _format_gsm8k(row)
            if formatted:
                sequences.append(formatted)
            if cfg.max_samples_per_source and i + 1 >= cfg.max_samples_per_source:
                break
    else:
        sequences.extend(_generate_synthetic_math(cfg.max_samples_per_source or 2000))

    code_ds = None
    for candidate in ("lvwerra/code_alpaca_20k", "codeparrot/codeparrot-clean", "the_stack_dedup"):
        code_ds = _safe_load_dataset(candidate, "train", cfg.dataset_cache_dir)
        if code_ds:
            break
    if code_ds:
        for i, row in enumerate(code_ds):
            formatted = _format_code(row)
            if formatted:
                sequences.append(formatted)
            if cfg.max_samples_per_source and i + 1 >= cfg.max_samples_per_source:
                break

    text_ds = _safe_load_dataset("wikitext", "train", cfg.dataset_cache_dir)
    if text_ds:
        for i, row in enumerate(text_ds):
            formatted = _format_wikitext(row)
            if formatted:
                sequences.append(formatted)
            if cfg.max_samples_per_source and i + 1 >= cfg.max_samples_per_source:
                break

    if not sequences:
        sequences.extend(_generate_synthetic_math(1000))
    random.shuffle(sequences)
    return sequences


class PackedDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, cfg: DataConfig, split: str = "train") -> None:
        self.seq_len = cfg.seq_len
        eos_id = tokenizer.token_to_id("<|eos|>")
        bos_id = tokenizer.token_to_id("<|bos|>")
        pad_id = tokenizer.token_to_id("<|pad|>")
        if eos_id is None or bos_id is None or pad_id is None:
            raise ValueError("Tokenizer missing required special tokens. Retrain tokenizer.")

        if cfg.packed_dataset_path and split == "train":
            data = torch.load(cfg.packed_dataset_path)
            self.inputs = data["inputs"]
            self.targets = data["targets"]
            return

        texts = _collect_text_samples(cfg)
        original_texts = list(texts)
        if original_texts:
            cutoff = max(1, int(0.9 * len(original_texts)))
            if split == "eval":
                texts = original_texts[cutoff:]
            else:
                texts = original_texts[:cutoff]
        if not texts:
            texts = original_texts
        tokens: List[int] = []
        for text in texts:
            encoded = tokenizer.encode(text)
            tokens.extend([bos_id])
            tokens.extend(encoded.ids)
            tokens.append(eos_id)
        # Ensure divisible by seq_len + 1 for target shift.
        block = self.seq_len + 1
        usable = (len(tokens) // block) * block
        tokens = tokens[:usable]
        if not tokens:
            raise RuntimeError("No tokenized samples available. Increase dataset limits or provide custom data.")
        tensor = torch.tensor(tokens, dtype=torch.long)
        chunks = tensor.view(-1, block)
        self.inputs = chunks[:, :-1].contiguous()
        self.targets = chunks[:, 1:].contiguous()
        if cfg.packed_dataset_path and split == "train":
            torch.save({"inputs": self.inputs, "targets": self.targets}, cfg.packed_dataset_path)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def build_dataloader(tokenizer: Tokenizer, data_cfg: DataConfig, train_cfg: TrainingConfig, split: str = "train") -> DataLoader:
    dataset = PackedDataset(tokenizer, data_cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size_per_gpu,
        shuffle=split == "train",
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
