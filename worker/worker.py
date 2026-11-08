"""Sharded teacher labelling worker.

Reads commands_pool.jsonl from S3, labels each command assigned to this shard
using Ollama's phi3.5 model, and writes results to datasets/teacher/shard-{i}.jsonl
in S3. Supports resuming from a partially written shard object.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SHARD_INDEX = int(os.environ.get("SHARD_INDEX", 0))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", 4))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
MODEL = os.environ.get("MODEL", "phi3.5")
PROMPT_FILE = os.environ.get("PROMPT_FILE", "/app/prompts/teacher_prompt.txt")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs.data.svc.cluster.local:8333")
CHECKPOINT = int(os.environ.get("CHECKPOINT", 20))

BUCKET = "datasets"
POOL_KEY = "commands_pool.jsonl"
SHARD_KEY = f"teacher/shard-{SHARD_INDEX}.jsonl"


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local-access-key"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local-secret-key"),
        region_name="us-east-1",
    )


def load_pool(s3):
    """Download the command pool from S3 or local file."""
    if os.path.exists("data/commands_pool.jsonl"):
        with open("data/commands_pool.jsonl", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        log.info("Loaded %d commands from local file", len(lines))
        return lines

    body = s3.get_object(Bucket=BUCKET, Key=POOL_KEY)["Body"].read().decode()
    lines = [json.loads(line) for line in body.strip().splitlines() if line.strip()]
    log.info("Pool has %d commands from S3", len(lines))
    return lines


def load_existing_shard(s3):
    """Try to load an existing shard to support resume."""
    try:
        body = s3.get_object(Bucket=BUCKET, Key=SHARD_KEY)["Body"].read().decode()
        lines = [line for line in body.strip().splitlines() if line.strip()]
        log.info("Resuming: found %d existing results in shard %d", len(lines), SHARD_INDEX)
        return lines
    except Exception:
        return []


def load_prompt():
    """Load the teacher prompt."""
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, encoding="utf-8") as f:
            return f.read().strip()
    if os.path.exists("prompts/teacher_prompt.txt"):
        with open("prompts/teacher_prompt.txt", encoding="utf-8") as f:
            return f.read().strip()
    return (
        'You are a command parser. Given a natural language command, output exactly one '
        'JSON object with four keys: "intent", "date", "time", "place". '
        'Output ONLY the JSON object.'
    )


def call_ollama(prompt, text, retries=3):
    """Send a request to Ollama and return the exact teacher reply string."""
    # Fixed decoding settings (temperature 0, seed 42) plus a JSON grammar and a
    # newline stop sequence, so phi3.5 emits exactly one object instead of
    # continuing with invented follow-up examples.
    payload = json.dumps({
        "model": MODEL,
        "prompt": f'Command: "{text}"',
        "system": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "seed": 42, "num_predict": 96, "stop": ["\n\n"]},
    }).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
                return result.get("response", "").strip()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            log.warning("Ollama attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Ollama failed after {retries} attempts")


def upload_shard(s3, lines):
    """Upload the shard content to S3."""
    body = "\n".join(lines) + "\n"
    s3.put_object(Bucket=BUCKET, Key=SHARD_KEY, Body=body.encode())


def main():
    log.info("Worker starting: shard %d of %d", SHARD_INDEX, SHARD_COUNT)
    s3 = get_s3()
    pool = load_pool(s3)
    prompt = load_prompt()

    # Filter lines where line_index % SHARD_COUNT == SHARD_INDEX
    my_lines = [(i, cmd) for i, cmd in enumerate(pool) if i % SHARD_COUNT == SHARD_INDEX]
    log.info("Shard %d has %d commands to label", SHARD_INDEX, len(my_lines))

    existing = load_existing_shard(s3)
    done_count = len(existing)
    results = list(existing)

    for idx, (line_idx, cmd) in enumerate(my_lines):
        if idx < done_count:
            continue

        text = cmd["text"]
        cmd_id = cmd["id"]

        try:
            raw_reply = call_ollama(prompt, text)
        except Exception as exc:
            log.error("Failed to label %s: %s", cmd_id, exc)
            raw_reply = ""

        # Strictly write {"id": "...", "text": "...", "teacher_raw": "<the teacher's exact reply>"}
        record = {
            "id": cmd_id,
            "text": text,
            "teacher_raw": raw_reply,
        }
        results.append(json.dumps(record))

        if (idx + 1) % CHECKPOINT == 0:
            log.info("Shard %d checkpoint at %d/%d", SHARD_INDEX, idx + 1, len(my_lines))
            upload_shard(s3, results)

    upload_shard(s3, results)
    log.info("Shard %d complete: %d results written", SHARD_INDEX, len(results))


if __name__ == "__main__":
    main()
