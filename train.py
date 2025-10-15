"""Standalone training script for the mini-GPT model."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from contextlib import contextmanager

from torch.utils.data import DataLoader

from config import ProjectConfig, load_default_config
from data import build_dataloader
from model import build_model
from tokenizer import load_tokenizer


LOGGER = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, base_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.base_lr = base_lr
        self.last_step = -1
        self.step()

    def step(self) -> None:
        self.last_step += 1
        lr = self.get_lr(self.last_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))

    def state_dict(self) -> Dict:
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.last_step = state_dict.get("last_step", 0)
        lr = self.get_lr(self.last_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int) -> Dict[str, float]:
    model.eval()
    losses = []
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(dataloader):
            if step >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets)
            losses.append(loss.item())
    model.train()
    if not losses:
        return {"loss": float("nan"), "ppl": float("nan")}
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "ppl": math.exp(mean_loss)}


def sample_outputs(model: torch.nn.Module, tokenizer, device: torch.device, prompts: Dict[str, str], cfg: ProjectConfig) -> Dict[str, str]:
    outputs = {}
    for name, text in prompts.items():
        encoded = tokenizer.encode(text)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
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


def save_checkpoint(output_dir: Path, step: int, model, optimizer, scheduler, scaler) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"model_step_{step}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return int(state.get("step", 0))


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def get_device(runtime_cfg) -> torch.device:
    if runtime_cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args: Optional[argparse.Namespace] = None) -> None:
    parser = argparse.ArgumentParser(description="Train the mini-GPT model from scratch.")
    parser.add_argument("--config-json", default=None, help="Path to JSON file with config overrides.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    cli_args = parser.parse_args(args=args)

    cfg = load_default_config()
    if cli_args.config_json:
        overrides = json.loads(Path(cli_args.config_json).read_text())
        cfg = cfg.override(**overrides)
    if cli_args.output_dir:
        cfg.runtime.output_dir = cli_args.output_dir
    if cli_args.resume:
        cfg.training.resume_from = cli_args.resume

    output_dir = Path(cfg.runtime.output_dir)
    setup_logging(output_dir)
    LOGGER.info("Starting run with config: %s", cfg)
    set_seed(cfg.runtime.seed)

    device = get_device(cfg.runtime)
    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    cfg.model.vocab_size = vocab_size

    model = build_model(cfg.model).to(device)
    if cfg.runtime.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_loader = build_dataloader(tokenizer, cfg.data, cfg.training, split="train")
    eval_loader = build_dataloader(tokenizer, cfg.data, cfg.training, split="eval")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, cfg.training.warmup_steps, cfg.training.total_steps, cfg.training.learning_rate)

    scaler = None
    amp_dtype = torch.float32
    if cfg.training.mixed_precision == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    elif cfg.training.mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    start_step = 0
    if cfg.training.resume_from:
        ckpt_path = Path(cfg.training.resume_from)
        if ckpt_path.exists():
            LOGGER.info("Resuming from checkpoint %s", ckpt_path)
            start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)

    total_steps = cfg.training.total_steps
    grad_accum = cfg.training.grad_accum_steps
    model.train()

    prompts = {
        "chat": "User: Hello! How are you today?\nAssistant:",
        "math": "User: Solve: 12 + 35\nAssistant:",
        "code": "User: Write a JavaScript function that reverses a string.\nAssistant:",
    }

    if device.type == "cuda":
        scaler_context = torch.cuda.amp.autocast  # type: ignore[attr-defined]
    else:
        @contextmanager
        def scaler_context(**_kwargs):
            yield

    global_step = start_step
    dataloader_iter = iter(train_loader)

    while global_step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for accum_step in range(grad_accum):
            try:
                inputs, targets = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_loader)
                inputs, targets = next(dataloader_iter)

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with scaler_context(dtype=amp_dtype):
                _, loss = model(inputs, targets)
                loss = loss / grad_accum
            accumulated_loss += loss.item()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if cfg.training.max_grad_norm:
            if scaler:
                scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        global_step += 1

        if global_step % cfg.training.log_interval == 0:
            LOGGER.info(
                "Step %d | loss %.4f | lr %.6f",
                global_step,
                accumulated_loss,
                optimizer.param_groups[0]["lr"],
            )

        if global_step % cfg.training.eval_interval == 0:
            metrics = evaluate(model, eval_loader, device, cfg.eval.max_eval_batches)
            LOGGER.info("Eval @ step %d | loss %.4f | ppl %.2f", global_step, metrics["loss"], metrics["ppl"])

        if global_step % cfg.training.sample_interval == 0:
            samples = sample_outputs(model, tokenizer, device, prompts, cfg)
            sample_path = output_dir / f"samples_step_{global_step}.txt"
            with sample_path.open("w") as f:
                for name, text in samples.items():
                    f.write(f"### {name}\n{text}\n\n")

        if global_step % cfg.training.checkpoint_interval == 0:
            ckpt_path = save_checkpoint(output_dir, global_step, model, optimizer, scheduler, scaler)
            LOGGER.info("Saved checkpoint to %s", ckpt_path)

    final_ckpt = save_checkpoint(output_dir, global_step, model, optimizer, scheduler, scaler)
    LOGGER.info("Training complete at step %d (final checkpoint: %s)", global_step, final_ckpt)


if __name__ == "__main__":
    main()
