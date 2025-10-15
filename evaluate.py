"""Evaluation script for the mini-GPT model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from config import load_default_config
from data import build_dataloader
from model import build_model
from tokenizer import load_tokenizer


PROMPTS = {
    "english": "Once upon a time,",
    "math": "Solve: (3x + 5 = 11)",
    "javascript": "Write a function add(a, b) {",
}


def generate_prompt_outputs(model, tokenizer, device: torch.device) -> Dict[str, str]:
    model.eval()
    outputs = {}
    for name, prompt in PROMPTS.items():
        encoding = tokenizer.encode(f"User: {prompt}\nAssistant:")
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
            )
        decoded = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
        outputs[name] = decoded
    return outputs


def compute_perplexity(model, dataloader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            if idx >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets)
            losses.append(loss.item())
    if not losses:
        return float("nan")
    mean_loss = sum(losses) / len(losses)
    return float(torch.exp(torch.tensor(mean_loss)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained mini-GPT model.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--max-eval-batches", type=int, default=50)
    args = parser.parse_args()

    cfg = load_default_config()
    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    cfg.model.vocab_size = tokenizer.get_vocab_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model, checkpoint_path=args.checkpoint, map_location=device)
    model.to(device)

    dataloader = build_dataloader(tokenizer, cfg.data, cfg.training, split="eval")
    perplexity = compute_perplexity(model, dataloader, device, args.max_eval_batches)

    outputs = generate_prompt_outputs(model, tokenizer, device)
    print("=== Prompt Outputs ===")
    for name, text in outputs.items():
        print(f"[{name}]\n{text}\n")
    print(f"Perplexity (approx): {perplexity:.3f}")


if __name__ == "__main__":
    main()
