from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interact with the fine-tuned chatbot")
    parser.add_argument(
        "--model_dir",
        default="/content/finetuned-chat-model",
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate per turn")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability")
    parser.add_argument("--history_turns", type=int, default=6, help="Number of previous turns to keep in context")
    return parser.parse_args()


def build_prompt(history: List[str], user_message: str, history_turns: int) -> str:
    trimmed_history = history[-history_turns * 2 :] if history_turns > 0 else []
    conversation = "\n".join(trimmed_history + [f"User: {user_message}", "Assistant:"])
    return conversation


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    history: List[str] = []

    print("Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye!")
            break

        prompt = build_prompt(history, user_message, args.history_turns)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        assistant_reply = decoded[len(prompt) :].strip()
        print(f"Assistant: {assistant_reply}")

        history.append(f"User: {user_message}")
        history.append(f"Assistant: {assistant_reply}")


if __name__ == "__main__":
    main()
