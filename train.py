"""
train.py — Train YOUR AI on any text you want!

USAGE:
  1. Put your training text in a file called  data.txt
  2. Run:  python train.py
  3. Wait (30 min - few hours depending on your PC)
  4. Your model saves to  my_ai.pt
"""

import torch
import pickle
import os
from model import MyGPT

# ─── SETTINGS (tweak these) ────────────────────────────────────────────────────
DATA_FILE    = "data.txt"       # Your training text
SAVE_FILE    = "my_ai.pt"       # Where to save your trained model
VOCAB_FILE   = "vocab.pkl"      # Character vocabulary

BATCH_SIZE   = 32               # How many samples per training step
BLOCK_SIZE   = 128              # How many characters the AI sees at once
EMBED_SIZE   = 256              # Model size (256=small, 512=medium)
NUM_HEADS    = 8                # Attention heads
NUM_LAYERS   = 6                # Transformer layers
MAX_ITERS    = 5000             # Training steps (more = smarter but slower)
LEARN_RATE   = 3e-4             # Learning rate
EVAL_EVERY   = 500              # Print loss every N steps
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️  Training on: {DEVICE.upper()}")
if DEVICE == "cpu":
    print("⚠️  No GPU found — training will be slow. Consider lowering MAX_ITERS to 2000.")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print(f"\n📖 Loading data from {DATA_FILE}...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    text = f.read()

print(f"   {len(text):,} characters loaded")

# Build character-level vocabulary
chars     = sorted(set(text))
vocab_size = len(chars)
stoi      = {c: i for i, c in enumerate(chars)}
itos      = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Save vocab for later (needed for the API)
with open(VOCAB_FILE, "wb") as f:
    pickle.dump({"stoi": stoi, "itos": itos}, f)

print(f"   Vocabulary size: {vocab_size} unique characters")

# Train/validation split
data   = torch.tensor(encode(text), dtype=torch.long)
n      = int(0.9 * len(data))
train  = data[:n]
val    = data[n:]

# ─── BATCH LOADER ─────────────────────────────────────────────────────────────
def get_batch(split):
    d   = train if split == "train" else val
    ix  = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([d[i:i+BLOCK_SIZE] for i in ix])
    y   = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ─── MODEL ────────────────────────────────────────────────────────────────────
model = MyGPT(
    vocab_size=vocab_size,
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_len=BLOCK_SIZE,
).to(DEVICE)

param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"\n🤖 Model created: {param_count:.1f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
print(f"\n🚀 Training for {MAX_ITERS} steps...\n")
best_val_loss = float("inf")

for step in range(MAX_ITERS):
    # Evaluate occasionally
    if step % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            train_loss = sum(model(*get_batch("train"))[1].item() for _ in range(20)) / 20
            val_loss   = sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20
        print(f"Step {step:5d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "embed_size": EMBED_SIZE,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "block_size": BLOCK_SIZE,
            }, SAVE_FILE)
            print(f"         ✅ Saved best model! (val loss: {val_loss:.4f})")
        model.train()

    # Training step
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print(f"\n🎉 Training complete! Your AI is saved to: {SAVE_FILE}")
print("👉 Now run:  python api.py  to start your personal AI API!")
