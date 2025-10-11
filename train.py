from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
from inspect import signature

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from data import DataConfig, load_mixed_dataset
from model import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-style model for conversational tasks")
    parser.add_argument("--model_name", default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--output_dir", default="/content/finetuned-chat-model", help="Directory to save the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length for inputs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Per-device eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging frequency in steps")
    parser.add_argument("--evaluation_strategy", default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"], help="Checkpoint save strategy")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional cap per dataset for debugging")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision if available")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision if available")
    parser.add_argument("--no_eval", action="store_true", help="Skip evaluation after training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Only one of --fp16 or --bf16 can be set.")

    fp16 = args.fp16 or (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7)
    bf16 = args.bf16 or (torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    torch_dtype = None
    if bf16:
        torch_dtype = torch.bfloat16
    elif fp16:
        torch_dtype = torch.float16

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=None,  # Trainer handles device placement
    )

    config = DataConfig(max_length=args.max_length, sample_size=args.sample_size)
    dataset = load_mixed_dataset(tokenizer, config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    raw_args = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "evaluation_strategy": "no" if args.no_eval else args.evaluation_strategy,
        "save_strategy": args.save_strategy,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "fp16": fp16 and not bf16,
        "bf16": bf16,
        "report_to": ["tensorboard"],
        "load_best_model_at_end": not args.no_eval,
        "save_total_limit": 2,
        "push_to_hub": False,
        "gradient_checkpointing": True,
    }

    supported_params = set(signature(TrainingArguments.__init__).parameters)
    filtered_args = {k: v for k, v in raw_args.items() if k in supported_params}

    if "evaluation_strategy" not in supported_params:
        filtered_args.pop("evaluation_strategy", None)
        if not args.no_eval:
            print("[train.py] Warning: installed transformers version does not support 'evaluation_strategy'; evaluation will be skipped.")
    if "save_strategy" not in supported_params:
        filtered_args.pop("save_strategy", None)
    if "load_best_model_at_end" not in supported_params:
        filtered_args.pop("load_best_model_at_end", None)
    if "gradient_checkpointing" not in supported_params:
        filtered_args.pop("gradient_checkpointing", None)
    if "report_to" not in supported_params:
        filtered_args.pop("report_to", None)

    training_args = TrainingArguments(**filtered_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None if args.no_eval else dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    if not args.no_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save data config for reproducibility.
    with open(os.path.join(args.output_dir, "data_config.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(asdict(config), f, indent=2)


if __name__ == "__main__":
    main()
