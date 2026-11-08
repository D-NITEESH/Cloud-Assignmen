"""Data loading utilities for training the command parser."""
import json


def load_jsonl(path):
    """Load a JSONL file and return a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_for_training(record):
    """Format a labelled record as an input-output training string.

    Input format required by specification:
    Command: <the command>
    JSON: <the teacher’s JSON reply>
    """
    label = json.dumps({
        "intent": record["intent"],
        "date": record.get("date"),
        "time": record.get("time"),
        "place": record.get("place"),
    })
    return f"Command: {record['text']}\nJSON: {label}"


def load_train_data(path, subset=None):
    """Load training data and format for the model.

    Args:
        path: Path to the JSONL training file.
        subset: If set, use only the first N records (for smoke tests).

    Returns:
        List of formatted training strings.
    """
    records = load_jsonl(path)
    if subset:
        records = records[:subset]
    return [format_for_training(r) for r in records]
