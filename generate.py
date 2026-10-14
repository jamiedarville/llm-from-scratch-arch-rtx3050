import torch
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("minigpt_shakespeare.pt", map_location=device, weights_only=False)

model = MiniGPT(
    vocab_size=ckpt["vocab_size"],
    n_embd=ckpt["n_embd"],
    n_head=ckpt["n_head"],
    n_layer=ckpt["n_layer"],
    block_size=ckpt["block_size"],
    dropout=0.0,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

stoi, itos = ckpt["stoi"], ckpt["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

prompt = "KING RICHARD III:\nWhat sayst thou, "
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

for temp in [0.5, 0.8, 1.2]:
    print(f"\n=== Temperature: {temp} ===")
    output_ids = model.generate(context, max_new_tokens=250, temperature=temp)[0].tolist()
    print(decode(output_ids))
