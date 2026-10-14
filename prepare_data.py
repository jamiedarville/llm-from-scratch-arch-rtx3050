with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Total dataset length: {len(text):,} characters")

# Extract all unique characters in the file
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters (vocab_size): {vocab_size}")
print("Vocabulary:", "".join(chars))

# Create a mapping from character to integer (stoi) and integer to character (itos)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

sample = "First Citizen:"
encoded = encode(sample)
print(f"\nOriginal text : {sample}")
print(f"Encoded tokens: {encoded}")
print(f"Decoded back  : {decode(encoded)}")
