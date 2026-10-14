# Building & Training an LLM from Scratch on Arch Linux (NVIDIA RTX 3050)

A hands-on, plain-English repository for learning the internal mechanics of a Decoder-Only GPT Transformer by building and training a **10.7M-parameter model** from scratch in [PyTorch](https://docs.pytorch.org/) on [Arch Linux](https://wiki.archlinux.org/title/NVIDIA) using an [NVIDIA GeForce RTX 3050](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050/).

## Hardware Overview: RTX 3050 Variants & Safe Defaults

| GPU Variant | VRAM & Bus | CUDA Cores | Safe Scratch Model Size | Recommended `batch_size` & `block_size` |
| :--- | :--- | :--- | :--- | :--- |
| [RTX 3050 Mobile](https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-mobile.c3788) | 4 GB GDDR6 (128-bit) | 2,048 | ~10M–15M parameters | `batch_size = 32`, `block_size = 128` |
| [RTX 3050 6 GB](https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-6-gb.c4188) | 6 GB GDDR6 (96-bit) | 2,304 | ~15M–25M parameters | `batch_size = 48`, `block_size = 256` |
| [RTX 3050 8 GB](https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-8-gb.c3858) | 8 GB GDDR6 (128-bit) | 2,560 | ~25M–45M parameters | `batch_size = 64`, `block_size = 256` |

## Step 1: Install NVIDIA Drivers, CUDA, and PyTorch on Arch Linux

```bash
sudo pacman -Syu
sudo pacman -S nvidia-open cuda cudnn nvtop btop git wget python python-pytorch-opt-cuda
```

*(If you use the `linux-lts` kernel, install `nvidia-open-lts`; for custom kernels like `linux-zen`, install `nvidia-open-dkms`. Reboot after installing drivers.)*

Create a virtual environment linked to Arch's optimized CUDA PyTorch package:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0), '| BF16:', torch.cuda.is_bf16_supported())"
```

## Step 2: Download the Dataset & Inspect Tokenization

Download Andrej Karpathy's [TinyShakespeare Dataset](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) (~1.1 MB) and run the character tokenizer inspection script:

```bash
wget -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python prepare_data.py
```

## Step 3: Train the 10.7M-Parameter Transformer (`train.py`)

Open a second terminal and run `nvtop` to monitor GPU utilization and VRAM usage in real time, then start training:

```bash
python train.py
```

* **Step 0 (`Loss ~4.17`)**: Random initialization babble ($-\ln(1/65) \approx 4.174$).
* **Step 300 (`Loss ~2.00`)**: Learns spaces, common syllables, and line breaks.
* **Step 1500 (`Loss ~1.55`)**: Learns speaker headers (`ROMEO:`) and verse structure.
* **Step 3000 (`Loss ~1.47`)**: Saves `minigpt_shakespeare.pt` and outputs coherent Shakespearean dialogue.

## Step 4: Generate Text at Different Temperatures (`generate.py`)

```bash
python generate.py
```

## Step 5: Four Experiments to Understand LLM Internals

1. **Switch to FlashAttention**: In `model.py` (`Head.forward`), replace manual attention score math with `F.scaled_dot_product_attention(q, k, v, is_causal=True)` to fuse operations in RTX 3050 SRAM (~2x speedup).
2. **Remove Residual Connections**: In `model.py` (`Block.forward`), change `x = x + self.sa(self.ln1(x))` to `x = self.sa(self.ln1(x))` to observe vanishing gradients across layers.
3. **Remove Positional Embeddings**: In `model.py` (`MiniGPT.forward`), change `x = tok_emb + pos_emb` to `x = tok_emb` to see why attention needs explicit position order.
4. **Compare Character vs. Subword BPE Tokenization**:
   ```bash
   pip install tiktoken
   python -c "import tiktoken; enc = tiktoken.get_encoding('gpt2'); s = 'First Citizen: Before we proceed any further, hear me speak.'; print('BPE tokens:', len(enc.encode(s)), 'vs Characters:', len(s))"
   ```
