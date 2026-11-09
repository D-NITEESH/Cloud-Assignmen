"""Post-processing: validate, deduplicate, and split teacher-labelled data.

Reads all shard outputs from S3 (or local directory), removes:
  (a) replies that are not valid JSON;
  (b) replies whose keys are not exactly intent, date, time, and place;
  (c) replies whose intent is not one of the eight;
  (d) replies with a slot value that is neither null nor text that appears in the command;
  (e) duplicate commands (keep the one with the smallest id).

Writes the cleaned dataset in id order to datasets/v1/train.jsonl,
and its SHA-256 checksum to datasets/v1/train.sha256.
Assorts and checks: input = (a) + (b) + (c) + (d) + duplicates + kept.
"""
import glob
import hashlib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
BUCKET = "datasets"
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", 4))

VALID_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query", "recommendation_events",
}
REQUIRED_KEYS = {"intent", "date", "time", "place"}


def get_s3():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local-access-key"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local-secret-key"),
        region_name="us-east-1",
    )


def load_shards():
    """Load records from local shard files or S3."""
    all_records = []

    # Check local directories first
    local_candidates = [
        "results/shards/*.jsonl",
        "data/shards/*.jsonl",
        "datasets/teacher/*.jsonl",
    ]
    for pattern in local_candidates:
        matched = sorted(glob.glob(pattern))
        if matched:
            log.info("Found local shard files matching %s: %s", pattern, matched)
            for fpath in matched:
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_records.append(json.loads(line))
            return all_records

    # Otherwise fetch from S3
    try:
        s3 = get_s3()
        for i in range(SHARD_COUNT):
            key = f"teacher/shard-{i}.jsonl"
            try:
                body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
                lines = [line for line in body.strip().splitlines() if line.strip()]
                log.info("Loaded shard %d from S3: %d lines", i, len(lines))
                for line in lines:
                    all_records.append(json.loads(line))
            except Exception as exc:
                log.warning("Could not read S3 key %s: %s", key, exc)
    except Exception as exc:
        log.warning("Could not connect to S3: %s", exc)

    return all_records


def parse_raw_response(raw):
    """Parse JSON from teacher raw output."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                obj = json.loads(part)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def clean_records(records):
    """Filter records by rules (a)-(e) and track exact counts."""
    counts = {
        "input": len(records),
        "a_invalid_json": 0,
        "b_wrong_keys": 0,
        "c_invalid_intent": 0,
        "d_slot_not_in_text": 0,
        "duplicates": 0,
        "kept": 0,
    }

    seen_texts = {}  # text -> record with smallest id
    valid_candidates = []

    for rec in records:
        raw = rec.get("teacher_raw")
        if raw is None:
            raw = rec.get("teacher_response")

        parsed = parse_raw_response(raw)

        # (a) replies that are not valid JSON
        if parsed is None or not isinstance(parsed, dict):
            counts["a_invalid_json"] += 1
            continue

        # (b) replies whose keys are not exactly intent, date, time, and place
        if set(parsed.keys()) != REQUIRED_KEYS:
            counts["b_wrong_keys"] += 1
            continue

        # (c) replies whose intent is not one of the eight
        if parsed["intent"] not in VALID_INTENTS:
            counts["c_invalid_intent"] += 1
            continue

        # (d) replies with a slot value that is neither null nor text that appears in command
        cmd_text = rec.get("text", "")
        cmd_text_lower = cmd_text.lower()
        slot_error = False

        for slot in ("date", "time", "place"):
            val = parsed.get(slot)
            if val is not None:
                if not isinstance(val, str):
                    slot_error = True
                    break
                if val.strip().lower() not in cmd_text_lower:
                    slot_error = True
                    break

        if slot_error:
            counts["d_slot_not_in_text"] += 1
            continue

        valid_candidates.append({
            "id": rec["id"],
            "text": cmd_text,
            "intent": parsed["intent"],
            "date": parsed["date"],
            "time": parsed["time"],
            "place": parsed["place"],
        })

    # (e) duplicate commands (keep the one with the smallest id)
    # Sort candidates by ID first
    valid_candidates.sort(key=lambda x: x["id"])

    kept = []
    seen_texts = set()

    for item in valid_candidates:
        norm_text = item["text"].strip().lower()
        if norm_text in seen_texts:
            counts["duplicates"] += 1
        else:
            seen_texts.add(norm_text)
            kept.append(item)

    counts["kept"] = len(kept)

    # Invariant assertion
    total_accounted = (
        counts["a_invalid_json"] +
        counts["b_wrong_keys"] +
        counts["c_invalid_intent"] +
        counts["d_slot_not_in_text"] +
        counts["duplicates"] +
        counts["kept"]
    )

    log.info(
        "Clean summary: input=%d, (a)=%d, (b)=%d, (c)=%d, (d)=%d, duplicates=%d, kept=%d (sum=%d)",
        counts["input"], counts["a_invalid_json"], counts["b_wrong_keys"],
        counts["c_invalid_intent"], counts["d_slot_not_in_text"],
        counts["duplicates"], counts["kept"], total_accounted
    )

    assert total_accounted == counts["input"], (
        f"Invariant broken: input {counts['input']} != sum {total_accounted}"
    )

    return kept, counts


def sha256_of_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    records = load_shards()
    if not records:
        log.warning("No records found in shards.")
        return

    kept, counts = clean_records(records)

    out_dir = "datasets/v1"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_path = os.path.join(out_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item) + "\n")

    checksum = sha256_of_file(train_path)
    sha_path = os.path.join(out_dir, "train.sha256")
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(checksum + "\n")

    counts["train_sha256"] = checksum
    counts_file = "results/clean_counts.json"
    with open(counts_file, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    # Upload to S3 if accessible
    try:
        s3 = get_s3()
        s3.upload_file(train_path, BUCKET, "v1/train.jsonl")
        s3.upload_file(sha_path, BUCKET, "v1/train.sha256")
        log.info("Uploaded train.jsonl and train.sha256 to s3://%s/v1/", BUCKET)
    except Exception as exc:
        log.warning("Could not upload to S3: %s", exc)

    print("\n=== Data Cleaning Summary ===")
    print(f"Total input records: {counts['input']}")
    print(f"  (a) Invalid JSON:  {counts['a_invalid_json']}")
    print(f"  (b) Wrong keys:    {counts['b_wrong_keys']}")
    print(f"  (c) Bad intent:    {counts['c_invalid_intent']}")
    print(f"  (d) Slot not in:   {counts['d_slot_not_in_text']}")
    print(f"  (e) Duplicates:    {counts['duplicates']}")
    print(f"Kept records:        {counts['kept']}")
    print(f"train.sha256:        {checksum}")


if __name__ == "__main__":
    main()
