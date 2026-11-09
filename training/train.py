"""Fine-tune EleutherAI/pythia-70m on teacher-labelled command data.

Usage:
    python training/train.py                          # full training
    python training/train.py --subset 50              # smoke test with 50 records
    python training/train.py --config training/config.yaml --version v2  # train v2
"""
import argparse
import csv
import os
import random

import torch
import yaml
from data_helpers import format_for_training, load_jsonl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class CommandDataset(Dataset):
    """Dataset of prompt/target token ids for causal LM training.

    Only the teacher's JSON reply contributes to the loss; the command prompt
    tokens are masked with -100 so the model learns to produce the reply.
    """

    def __init__(self, texts, tokenizer, max_length):
        self.items = []
        eos_id = tokenizer.eos_token_id
        for text in texts:
            prompt, target = text.split("\nJSON:", 1)
            prompt_ids = tokenizer(prompt + "\nJSON:", add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [eos_id]
            input_ids = (prompt_ids + target_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
            self.items.append((input_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        input_ids, labels = self.items[idx]
        return {"input_ids": input_ids, "labels": labels}


def collate_batch(batch, pad_token_id):
    """Pad a batch to its own longest sequence (dynamic padding)."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[row, :n] = torch.tensor(item["input_ids"])
        attention_mask[row, :n] = 1
        labels[row, :n] = torch.tensor(item["labels"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train(config, subset=None, version="v1"):
    """Run training loop."""
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    model_name = config["model_name"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    max_length = config["max_length"]
    train_file = config["train_file"]

    if version == "v2":
        output_dir = config.get("output_dir", "models/v1").replace("v1", "v2")
        loss_file = config.get("loss_file", "training/loss.csv").replace("loss.csv", "loss_v2.csv")
    else:
        output_dir = config.get("output_dir", "models/v1")
        loss_file = config.get("loss_file", "training/loss.csv")

    # A smoke run must not overwrite the real artefacts.
    if subset:
        output_dir = "models/subset-test"
        loss_file = "training/loss_subset.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Load data
    print(f"Loading training data from: {train_file}")
    records = load_jsonl(train_file)
    if subset:
        records = records[:subset]
        print(f"Using subset of {subset} records")
    else:
        # Assignment limit: at most the first 1,500 cleaned records.
        n_train = int(config.get("n_train", 1500))
        records = records[:n_train]
        print(f"Using first {len(records)} records (n_train={n_train})")

    # For v2, use the second half of the data for slight variation
    if version == "v2":
        mid = len(records) // 2
        records = records[mid:] + records[:mid]
        print(f"v2: rotated data, starting from record {mid}")

    texts = [format_for_training(r) for r in records]
    print(f"Training on {len(texts)} examples")

    dataset = CommandDataset(texts, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    losses = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append({"epoch": epoch, "loss": round(avg_loss, 6)})
        print(f"Epoch {epoch}/{epochs} — loss: {avg_loss:.6f}")

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Record a SHA-256 checksum for every saved artefact (rubric: models/vN/model.sha256)
    import hashlib
    with open(os.path.join(output_dir, "model.sha256"), "w", encoding="utf-8") as f:
        for name in sorted(os.listdir(output_dir)):
            if name == "model.sha256":
                continue
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
                f.write(f"{digest}  {name}\n")

    print(f"Model saved to {output_dir}")

    # Save loss CSV
    os.makedirs(os.path.dirname(loss_file), exist_ok=True)
    with open(loss_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(losses)
    print(f"Loss saved to {loss_file}")


def main():
    parser = argparse.ArgumentParser(description="Train command parser model")
    parser.add_argument("--config", default="training/config.yaml", help="Config file path")
    parser.add_argument("--subset", type=int, default=None, help="Use only first N records")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"], help="Model version")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train(config, subset=args.subset, version=args.version)


if __name__ == "__main__":
    main()
