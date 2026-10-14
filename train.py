import time
import torch
from model import MiniGPT

batch_size = 64        # Lower to 32 on a 4GB RTX 3050 if needed
block_size = 256       # Context window length
max_iters = 3000       # Total training steps
eval_interval = 300    # How often to check validation loss
learning_rate = 3e-4   # Standard AdamW learning rate
eval_iters = 100       # Batches averaged for loss evaluation
n_embd = 384           # Embedding dimension
n_head = 6             # Number of attention heads
n_layer = 6            # Number of Transformer blocks
dropout = 0.2          # Dropout probability
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Initialized MiniGPT on {device} with {num_params / 1e6:.2f}M parameters")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n[Untrained Model Output (Step 0)]:")
print(decode(model.generate(context, max_new_tokens=150)[0].tolist()))
print("-" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_time = time.time()

for iter_num in range(max_iters + 1):
    if iter_num % eval_interval == 0:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(
            f"Step {iter_num:4d} | Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f} | VRAM: {vram_gb:.2f} GB | Time: {elapsed:.1f}s"
        )

    xb, yb = get_batch("train")

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,
    "stoi": stoi,
    "itos": itos,
}
torch.save(checkpoint, "minigpt_shakespeare.pt")
print("\nSaved checkpoint to minigpt_shakespeare.pt")

model.eval()
print("\n[Trained Model Output (Step 3000)]:")
print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))
