"""Interactive sampling script for chatting with the trained mini-GPT model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from config import load_default_config
from model import build_model
from tokenizer import load_tokenizer


def chat_loop(model, tokenizer, device: torch.device, max_new_tokens: int, temperature: float, top_k: Optional[int], top_p: Optional[float]) -> None:
    model.eval()
    print("Type 'exit' to quit the chat.")
    while True:
        user_input = input("user> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("assistant> Goodbye!")
            break
        prompt = f"User: {user_input}\nAssistant:"
        encoding = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        response = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
        assistant_reply = response.split("Assistant:", 1)[-1].strip()
        print(f"assistant> {assistant_reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interact with a trained mini-GPT checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    cfg = load_default_config()
    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    cfg.model.vocab_size = tokenizer.get_vocab_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model, checkpoint_path=args.checkpoint, map_location=device)
    model.to(device)

    chat_loop(model, tokenizer, device, args.max_new_tokens, args.temperature, args.top_k, args.top_p)


main()
