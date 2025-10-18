"""Utilities for training or loading a byte-level BPE tokenizer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from config import load_default_config


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


def iter_training_texts(sources: List[str], limit_per_source: int | None = None) -> Iterable[str]:
    # Import inside function to avoid circular import when other local modules
    # also import HF `datasets` as `hf_datasets`.
    from datasets import load_dataset as hf_load_dataset

    for source in sources:
        try:
            dataset = hf_load_dataset(source, split="train")
        except Exception:
            continue
        count = 0
        for row in dataset:
            text = None
            if isinstance(row, dict):
                for key in ("text", "content", "instruction", "prompt"):
                    if key in row and isinstance(row[key], str):
                        text = row[key]
                        break
            if not text:
                continue
            yield text
            count += 1
            if limit_per_source and count >= limit_per_source:
                break


def train_tokenizer(output_path: str, vocab_size: int = 32000, sample_sources: List[str] | None = None, limit_per_source: int | None = 10_000) -> Tokenizer:
    sample_sources = sample_sources or ["wikitext", "openwebtext"]
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter_training_texts(sample_sources, limit_per_source), trainer=trainer)
    tokenizer.post_processor = None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    return Tokenizer.from_file(output_path)


def load_tokenizer(path: str) -> Tokenizer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found at {path}. Run tokenizer training first.")
    return Tokenizer.from_file(path)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Train or inspect the project tokenizer.")
    parser.add_argument("--output", default=None, help="Path to save tokenizer JSON.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--sources", nargs="*", default=None, help="HF dataset names for tokenizer training samples.")
    parser.add_argument("--limit", type=int, default=10000, help="Max samples per source.")
    parser.add_argument("--print-info", action="store_true", help="Print tokenizer statistics without training.")
    args = parser.parse_args()

    cfg = load_default_config()
    output_path = args.output or cfg.data.tokenizer_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if args.print_info and os.path.exists(output_path):
        tokenizer = load_tokenizer(output_path)
        info = {
            "vocab_size": tokenizer.get_vocab_size(),
            "special_tokens": SPECIAL_TOKENS,
            "path": output_path,
        }
        print(json.dumps(info, indent=2))
        return

    tokenizer = train_tokenizer(output_path=output_path, vocab_size=args.vocab_size, sample_sources=args.sources, limit_per_source=args.limit)
    print(f"Saved tokenizer to {output_path} (vocab {tokenizer.get_vocab_size()}).")


if __name__ == "__main__":
    cli()
