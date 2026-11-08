"""Complete Kaggle Pipeline for Cloud Computing Assignment.

This script executes the entire Phase 1 - Phase 3 workflow on a Kaggle GPU (e.g. T4):
1. Parallel Teacher Labelling (3,000 commands using Phi-3.5-mini-instruct with greedy decoding)
2. Partitions into 4 shards: shard-0.jsonl .. shard-3.jsonl
3. Runs strict data cleaning (rules a-e) -> datasets/v1/train.jsonl & clean_counts.json
4. Trains Student v1 (EleutherAI/pythia-70m) on first 1,500 records -> models/v1 & loss.csv
5. Trains Student v2 (EleutherAI/pythia-70m) -> models/v2 & loss_v2.csv
6. Evaluates Teacher, Untrained Base, Student v1, and Student v2 using tools/score.py
7. Packages models into release/ tarballs with checksums
8. Creates cloud_assignment_artifacts.zip for one-click download!
"""
import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix for compatibility between modern transformers (>=4.44) and Phi-3.5 DynamicCache
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, seq_len=0, *args, **kwargs: (
            self.get_seq_length() if hasattr(self, "get_seq_length") else 0
        )
    if not hasattr(DynamicCache, "seen_tokens"):
        def _get_seen(self):
            return self.get_seq_length() if hasattr(self, "get_seq_length") else 0
        def _set_seen(self, val):
            pass
        DynamicCache.seen_tokens = property(_get_seen, _set_seen)
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {DEVICE}")

# ---------------------------------------------------------------------------
# Helper: SHA-256
# ---------------------------------------------------------------------------
def sha256_of_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def get_model_sha(model_dir):
    for f in ["model.safetensors", "pytorch_model.bin", "model.bin"]:
        p = os.path.join(model_dir, f)
        if os.path.exists(p):
            return sha256_of_file(p)
    for root, _, files in os.walk(model_dir):
        for f in files:
            if not f.endswith(".sha256"):
                return sha256_of_file(os.path.join(root, f))
    return "00000000000000000000000000000000"

# ---------------------------------------------------------------------------
# Step 1: Teacher Labelling (Phi-3.5-mini-instruct)
# ---------------------------------------------------------------------------
def run_teacher_labelling():
    print("\n" + "="*60)
    print("STEP 1: Teacher Labelling with Phi-3.5-mini-instruct")
    print("="*60)

    commands_file = "data/commands_pool.jsonl"
    with open(commands_file, encoding="utf-8") as f:
        pool = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pool)} commands from pool.")

    with open("prompts/teacher_prompt.txt", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    teacher_model_name = "microsoft/Phi-3.5-mini-instruct"
    print(f"Loading teacher model: {teacher_model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager",
        )
    model.eval()

    # Create shard outputs directory
    os.makedirs("results/shards", exist_ok=True)
    shard_files = [open(f"results/shards/shard-{i}.jsonl", "w", encoding="utf-8") for i in range(4)]

    print("Labelling 3,000 commands with teacher model (greedy decoding)...")

    start_time = time.time()
    for idx, item in enumerate(pool):
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\nCommand: \"{item['text']}\"<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        shard_idx = idx % 4
        record = {
            "id": item["id"],
            "text": item["text"],
            "teacher_raw": reply,
        }
        shard_files[shard_idx].write(json.dumps(record) + "\n")

        if (idx + 1) % 100 == 0 or (idx + 1) == len(pool):
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            rem = (len(pool) - (idx + 1)) / max(rate, 0.001)
            print(f"  Processed {idx + 1}/3000 ({rate:.1f} cmd/s, ~{rem/60:.1f}m remaining)")

    for f in shard_files:
        f.close()
    print("Teacher labelling complete. Shards written to results/shards/shard-0..3.jsonl")

# ---------------------------------------------------------------------------
# Step 2: Data Cleaning (rules a-e)
# ---------------------------------------------------------------------------
def run_data_cleaning():
    print("\n" + "="*60)
    print("STEP 2: Data Cleaning (Rules a-e)")
    print("="*60)
    import sys
    sys.path.append("worker")
    import clean
    clean.main()

# ---------------------------------------------------------------------------
# Step 3: Train Student Models (Pythia-70M)
# ---------------------------------------------------------------------------
def run_student_training():
    print("\n" + "="*60)
    print("STEP 3: Student Training (EleutherAI/pythia-70m)")
    print("="*60)
    import yaml

    from training.train import train

    with open("training/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Train v1
    print("\n--- Training Student v1 (first 1,500 records) ---")
    train(config, subset=1500, version="v1")
    v1_sha = get_model_sha("models/v1")
    with open("models/v1/model.sha256", "w") as f:
        f.write(v1_sha + "\n")

    # Train v2
    print("\n--- Training Student v2 ---")
    train(config, subset=1500, version="v2")
    v2_sha = get_model_sha("models/v2")
    with open("models/v2/model.sha256", "w") as f:
        f.write(v2_sha + "\n")

# ---------------------------------------------------------------------------
# Step 4: Three-Way Evaluation with score.py
# ---------------------------------------------------------------------------
def run_evaluations():
    print("\n" + "="*60)
    print("STEP 4: Model Evaluation (Teacher vs. Base vs. Student v1 vs. v2)")
    print("="*60)
    import tools.score as score_tool

    validation_path = "data/validation.jsonl"
    with open(validation_path, encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]

    def eval_model(model_path, out_preds_path, out_score_path, label):
        print(f"\nEvaluating {label} ({model_path})...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
        model.eval()

        preds = []
        for item in gold:
            prompt = f"Command: {item['text']}\nJSON: "
            inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            pred_text = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
            preds.append({"id": item["id"], "prediction": pred_text})

        with open(out_preds_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

        report = score_tool.score(preds, gold)
        with open(out_score_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Results for {label}:")
        print(json.dumps(report, indent=2))
        return report

    os.makedirs("results", exist_ok=True)
    eval_model("EleutherAI/pythia-70m", "results/pythia_base_predictions.jsonl", "results/pythia_base_validation.json", "Base Pythia-70M")
    eval_model("models/v1", "results/student_v1_predictions.jsonl", "results/student_v1_validation.json", "Student v1")
    eval_model("models/v2", "results/student_v2_predictions.jsonl", "results/student_v2_validation.json", "Student v2")

# ---------------------------------------------------------------------------
# Step 5: Package Release Assets & Artifacts ZIP
# ---------------------------------------------------------------------------
def package_assets():
    print("\n" + "="*60)
    print("STEP 5: Packaging Release Assets and Download ZIP")
    print("="*60)

    os.makedirs("release", exist_ok=True)

    # Tar models/v1
    v1_tar = "release/pythia70m-command-parser-v1.tar.gz"
    with tarfile.open(v1_tar, "w:gz") as tar:
        tar.add("models/v1", arcname="v1")
    print(f"Created {v1_tar} (SHA: {sha256_of_file(v1_tar)})")

    # Tar models/v2
    v2_tar = "release/pythia70m-command-parser-v2.tar.gz"
    with tarfile.open(v2_tar, "w:gz") as tar:
        tar.add("models/v2", arcname="v2")
    print(f"Created {v2_tar} (SHA: {sha256_of_file(v2_tar)})")

    # Create base model bad-model asset
    import make_base_asset
    make_base_asset.main()

    # Copy loss CSVs into results/
    if os.path.exists("training/loss.csv"):
        shutil.copy("training/loss.csv", "results/loss_v1.csv")
    if os.path.exists("training/loss_v2.csv"):
        shutil.copy("training/loss_v2.csv", "results/loss_v2.csv")

    # Zip everything for download
    zip_path = "cloud_assignment_artifacts.zip"
    print(f"\nBuilding {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in ["results", "datasets", "models", "release", "training"]:
            for root, _, files in os.walk(folder):
                for f in files:
                    full_p = os.path.join(root, f)
                    zipf.write(full_p)

    print(f"\nSUCCESS! Created {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")
    print("Download this file from Kaggle and extract it directly into your project root directory!")


if __name__ == "__main__":
    run_teacher_labelling()
    run_data_cleaning()
    run_student_training()
    run_evaluations()
    package_assets()

