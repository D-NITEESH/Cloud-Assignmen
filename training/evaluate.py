"""Evaluate a trained model on the validation set.

Usage:
    python training/evaluate.py --model models/v1 --out results/student_v1_validation.json
    python training/evaluate.py --teacher --out results/teacher_validation.json
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_student(model_dir, validation_path, max_new_tokens=96):
    """Run inference with a fine-tuned student model."""
    print(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    predictions = []
    with open(validation_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Must match training/data_helpers.format_for_training exactly.
            prompt = f"Command: {record['text']}\nJSON:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the generated part (after the prompt)
            prediction = full_text[len(prompt):].strip()
            predictions.append({
                "id": record["id"],
                "prediction": prediction,
            })
            print(f"  {record['id']}: {prediction[:80]}")

    return predictions


def run_teacher(validation_path, ollama_url="http://localhost:11434", model="phi3.5"):
    """Run inference with the teacher model (Ollama)."""
    import urllib.error
    import urllib.request

    # Load prompt
    prompt_path = "prompts/teacher_prompt.txt"
    if os.path.exists(prompt_path):
        with open(prompt_path, encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = (
            'You are a command parser. Output exactly one JSON object with keys: '
            '"intent", "date", "time", "place".'
        )

    predictions = []
    with open(validation_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Identical decoding settings to worker/worker.py.
            payload = json.dumps({
                "model": model,
                "prompt": f'Command: "{record["text"]}"',
                "system": system_prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "seed": 42, "num_predict": 96,
                            "stop": ["\n\n"]},
            }).encode()

            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode())
                    prediction = result.get("response", "").strip()
            except Exception as exc:
                print(f"  {record['id']}: ERROR - {exc}")
                prediction = ""

            predictions.append({
                "id": record["id"],
                "prediction": prediction,
            })
            print(f"  {record['id']}: {prediction[:80]}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate command parser model")
    parser.add_argument("--model", default="models/v1", help="Path to model directory")
    parser.add_argument("--validation", default="data/validation.jsonl", help="Validation data")
    parser.add_argument("--out", default="results/predictions.json", help="Output predictions file")
    parser.add_argument("--teacher", action="store_true", help="Use teacher model (Ollama)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Max tokens to generate")
    args = parser.parse_args()

    if args.teacher:
        predictions = run_teacher(args.validation, args.ollama_url)
    else:
        predictions = run_student(args.model, args.validation, args.max_new_tokens)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
