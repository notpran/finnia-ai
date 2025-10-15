# mini-GPT Scratch Project

This repo contains a modular, Colab-friendly pipeline for training a small conversational transformer **from scratch**. The model targets a "mini-GPT" use case: lightweight dialogue, stepwise math reasoning, and JavaScript snippet generation. It is **not** meant to rival large frontier models—training from scratch beyond toy scales demands massive compute and data. This project instead focuses on clear structure, reproducibility, and knobs that let you scale up when resources grow.

## Reality Check
- Expect the default configuration to train in a single Google Colab GPU session (2-6 hours) and deliver a very modest model.
- True GPT-quality models require billions of tokens, long schedules, and expensive hardware. Use this project as a learning scaffold or seed for later fine-tuning.
- Datasets referenced here are public but often large; code samples cap downloads with per-source limits and include synthetic fallbacks so short runs stay lightweight.

## Project Layout
- `config.py` – Central dataclasses for model, training, data, and runtime settings (override-friendly).
- `model.py` – Pure PyTorch GPT-style transformer (embeddings, causal attention, MLP, generation helper).
- `tokenizer.py` – Byte-level BPE training/loading via `tokenizers`; re-trainable vocab, special tokens baked in.
- `datasets.py` – Dataset loaders and packing utilities. Fetches open chat, math, code, and text sources; falls back to synthetic arithmetic when needed.
- `train.py` – Full training loop with gradient accumulation, mixed precision, warmup+cosine LR scheduler, checkpointing, and periodic sampling.
- `sample.py` – Simple chat REPL that loads a checkpoint and responds with temperature/top-k/top-p sampling.
- `evaluate.py` – Scripted prompts plus held-out perplexity estimation for quick health checks.
- `artifacts/` (created at runtime) – Tokenizer JSON, cached datasets, checkpoints, logs, and sample generations.

## Quickstart (Google Colab)
1. **Create a new notebook** and enable GPU (`Runtime` → `Change runtime type` → `GPU`).
2. **Clone & install dependencies:**
   ```bash
   !git clone https://github.com/notpran/finnia-ai.git
   %cd finnia-ai
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   !pip install datasets tokenizers transformers accelerate tqdm numpy
   ```
3. **Train tokenizer (one-time or when changing vocab size):**
   ```bash
   !python tokenizer.py --vocab-size 32000 --sources wikitext openwebtext --limit 5000
   ```
4. **Kick off training:**
   ```bash
   !python train.py --output-dir artifacts/run_colab
   ```
   Monitor progress in the streamed logs (`loss`, learning rate, eval metrics). Checkpoints and sample generations land under `artifacts/run_colab/`.
5. **Chat with your model:**
   ```bash
   !python sample.py --checkpoint artifacts/run_colab/checkpoints/model_step_20000.pt
   ```
6. **Run quick evaluations:**
   ```bash
   !python evaluate.py --checkpoint artifacts/run_colab/checkpoints/model_step_20000.pt
   ```

## Configuration Tips
- Override settings by editing `config.py` or passing a JSON file:
  ```bash
  !python train.py --config-json configs/colab_small.json
  ```
  Example overrides:
  ```json
  {
    "model.n_layers": 8,
    "model.d_model": 640,
    "training.total_steps": 40000,
    "data.max_samples_per_source": 50000
  }
  ```
- **Scaling up:** increase layers, heads, `d_model`, dataset limits, and training steps. Expect memory growth roughly quadratic in `d_model` and linear in `seq_len`.
- **Scaling down:** reduce `seq_len` to 256, trim dataset limits, or drop to four layers to run on smaller GPUs.
- Enable `torch.compile` by setting `runtime.use_compile = True` for potential speedups (PyTorch 2.0+).
- Switch mixed precision modes (`fp16`, `bf16`, `fp32`) via `training.mixed_precision` if your GPU dictates.

## Data Notes
- Primary sources: `OpenAssistant/oasst1` (chat), `gsm8k` (math), `lvwerra/code_alpaca_20k` (code), `wikitext` (fluency).
- The loader caps each source at `data.max_samples_per_source` (default 20k). Adjust upward for better coverage when compute allows.
- Synthetic arithmetic is generated if real datasets fail to download, keeping short runs functional offline.

## Training Features
- Gradient accumulation simulates larger batch sizes without overloading VRAM (default effective batch: 32).
- `WarmupCosineScheduler` performs linear warmup then cosine decay over `training.total_steps`.
- Mixed precision via `torch.cuda.amp` speeds up training and lowers memory usage when GPUs support it.
- Checkpoints include model weights, optimizer, scheduler, AMP scaler, and step for seamless resume (`--resume path/to/checkpoint.pt`).
- Every `sample_interval` steps the trainer writes prompt completions (`samples_step_*.txt`) so you can track qualitative progress.

## Evaluation
- `evaluate.py` prints generations for:
  - English fluency: `Once upon a time,`
  - Math reasoning: `Solve: (3x + 5 = 11)`
  - JavaScript: `Write a function add(a, b) {`
- Also reports approximate perplexity on a held-out slice assembled by the dataset loader.

## Scaling Beyond Colab
1. **Multi-GPU:** wrap the training loop with `accelerate` or PyTorch DDP. The code toggles `runtime.use_accelerate` for future integration.
2. **Longer schedules:** extend `training.total_steps`, ensure dataset limits are high enough to avoid memorization, and monitor overfitting via perplexity.
3. **Larger vocab / context:** re-run `tokenizer.py` with bigger `--vocab-size` and raise `model.seq_len`. Expect slower training and higher memory use.
4. **Higher precision:** if instability appears in `fp16`, switch to `bf16` (Ampere+) or `fp32` for stability at the cost of throughput.

## Reproducibility & Logging
- All scripts print the active configuration and seed at startup.
- Training logs (`artifacts/run/train.log`) capture metrics and checkpoints produced.
- Random seeds are controlled via `config.py` but perfect determinism is not guaranteed on all GPU kernels.

## Next Steps
- Extend `datasets.py` with additional domains (e.g., reasoning, multilingual corpora) or plug in your own HF datasets.
- Swap the tokenizer for a pretrained one (e.g., `gpt2`) by editing `tokenizer.py` and aligning vocab size in `config.py`.
- Integrate LoRA or adapter layers for fine-tuning once a base model is trained.

Happy experimenting, and remember: meaningful capability jumps come from more data, more steps, and more compute—scale thoughtfully!
