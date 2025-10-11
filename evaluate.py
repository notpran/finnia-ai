from __future__ import annotations

import argparse
from textwrap import dedent
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SAMPLE_PROMPTS: List[str] = [
    "User: What are the three laws of motion?\nAssistant:",
    dedent(
        """User: Solve 24x + 13 = 85. Show each step.\nAssistant:"""
    ).strip(),
    dedent(
        """User: Write a JavaScript function called sumArray(arr) that returns the sum of an array. Explain the code.\nAssistant:"""
    ).strip(),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned chatbot on sample prompts")
    parser.add_argument(
        "--model_dir",
        default="/content/finetuned-chat-model",
        help="Directory containing the fine-tuned model and tokenizer",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    return parser.parse_args()


def generate_responses(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    responses: List[str] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        responses.append(decoded[len(prompt) :].strip())
    return responses


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    responses = generate_responses(
        tokenizer,
        model,
        SAMPLE_PROMPTS,
        device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    for prompt, response in zip(SAMPLE_PROMPTS, responses):
        print("=" * 40)
        print(prompt)
        print("Assistant:")
        print(response)
        print()


if __name__ == "__main__":
    main()
