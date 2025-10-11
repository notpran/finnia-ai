# Finetuned Conversational Transformer

This project fine-tunes a GPT-style language model (GPT-2 or EleutherAI/gpt-neo-125M) into a multi-domain assistant that can:

- Hold short English conversations.
- Solve math problems with step-by-step reasoning.
- Write and explain JavaScript code.

The workflow targets Google Colab with GPU acceleration and uses Hugging Face Transformers, Datasets, and PyTorch.

## Repository Structure

- `model.py` – Loads a pretrained causal language model and tokenizer, configuring padding tokens.
- `data.py` – Builds a mixed dataset from OpenAssistant, GSM8K, and CodeAlpaca formatted as dialogue pairs.
- `train.py` – Fine-tunes the model using the Hugging Face Trainer API.
- `chat.py` – Interactive REPL for chatting with the fine-tuned model.
- `evaluate.py` – Runs a lightweight evaluation on sample prompts spanning conversation, math, and coding.
- `data/` – Placeholder directory if you want to cache datasets locally (optional).

## Quickstart on Google Colab

1. **Open a new Colab notebook** and select **Runtime → Change runtime type → GPU**.
2. **Clone the repo and install dependencies:**

```python
!git clone https://github.com/your-username/finnia-ai.git
%cd finnia-ai
!pip install -U accelerate transformers datasets peft torch evaluate
```

3. **(Optional) Mount Google Drive** if you want to persist checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. **Launch training:**

```python
!python train.py \
  --model_name EleutherAI/gpt-neo-125M \
  --output_dir /content/finetuned-chat-model \
  --max_length 512 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --epochs 2 \
  --learning_rate 5e-5 \
  --warmup_steps 200 \
  --fp16
```

Adjust hyperparameters or add `--sample_size 1000` for faster debugging.

5. **Run interactive chat after training:**

```python
!python chat.py --model_dir /content/finetuned-chat-model
```

6. **Evaluate on sample prompts:**

```python
!python evaluate.py --model_dir /content/finetuned-chat-model
```

## Dataset Notes

- **OpenAssistant/oasst1** provides human–assistant chat pairs.
- **gsm8k** contributes math problems; the preprocessing prepends "Let's reason step by step." to encourage chain-of-thought outputs.
- **fka/CodeAlpaca-20k** supplies coding instructions and completions.

Datasets are automatically downloaded via the Hugging Face Datasets library and combined into a single dialogue-style corpus with padding/truncation to a fixed length.

## Tips

- Enable **mixed precision** (`--fp16` or `--bf16`) when using GPUs that support it.
- Use `--sample_size` to cap each dataset during experimentation.
- To resume training, re-run `train.py` with `--output_dir` pointing to the existing checkpoint and remove `--overwrite_output_dir`.

## License

This project depends on third-party datasets and models; respect their original licenses when redistributing checkpoints.
