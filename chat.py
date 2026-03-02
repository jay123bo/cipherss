"""
chat.py — Talk to YOUR AI right in the terminal!
No internet needed. 100% yours.
"""

import torch
import pickle
from model import MyGPT

MODEL_FILE = "my_ai.pt"
VOCAB_FILE  = "vocab.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n  Loading your AI brain...")

try:
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
except FileNotFoundError:
    print("\n  ERROR: AI not trained yet! Run SETUP.bat first.\n")
    exit()

stoi = vocab["stoi"]
itos = vocab["itos"]

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: "".join([itos.get(i, "?") for i in l])

model = MyGPT(
    vocab_size=checkpoint["vocab_size"],
    embed_size=checkpoint["embed_size"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    max_len=checkpoint["block_size"],
).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("  Your AI is ready!\n")
print("  " + "─" * 40)
print("  Type anything and press Enter.")
print("  Type 'quit' to exit.")
print("  " + "─" * 40 + "\n")

while True:
    try:
        prompt = input("  You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  Goodbye!\n")
        break

    if not prompt:
        continue
    if prompt.lower() in ("quit", "exit", "bye"):
        print("\n  Goodbye!\n")
        break

    # Generate response
    encoded = torch.tensor([encode(prompt)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            encoded,
            max_new_tokens=200,
            temperature=0.8,
            top_k=40
        )

    # Only show the new part
    new_tokens = output[0][len(encoded[0]):].tolist()
    response = decode(new_tokens).strip()

    # Clean up the response a bit
    if "\n" in response:
        response = response.split("\n")[0].strip()

    print(f"\n  Cipher.AI: {response}\n")
