# finnia ai

This is the open source code used to train finnia ai, a very tiny llm i made(dumb asf)

## Reality Check
- this model is maybe an llm but its really dumb, like really really dumb :/

## Quickstart
1. **Clone & install dependencies:**
   ```bash
   git clone https://github.com/notpran/finnia-ai.git
   cd finnia-ai
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install datasets tokenizers transformers accelerate tqdm numpy
   ```
3. **Train tokenizer (one-time or when changing vocab size):**
   ```bash
   !python tokenizer.py --vocab-size 32000 --sources wikitext openwebtext --limit 5000
   ```
4. **training**
   ```bash
   !python train.py --output-dir artifacts/run_colab
   ```

5. **Chat with your model:**
   ```bash
   !python sample.py --checkpoint artifacts/run_colab/checkpoints/model_step_20000.pt
   ```
