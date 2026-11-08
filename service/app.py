"""FastAPI command parser service.

Endpoints:
    GET  /healthz  — returns 200 as long as process is alive
    GET  /readyz   — 503 until model is loaded, 200 after that
    POST /parse    — parse command into 4-key JSON

Response headers:
    X-Served-By: hostname (pod name)
    X-Model-Version: version of the loaded model

Environment variables:
    MODEL_URI: path or S3 URI (e.g., s3://models/v1)
    MODEL_VERSION: version string (e.g., v1)
    S3_ENDPOINT: S3 endpoint URL
    MAX_NEW_TOKENS: maximum tokens to generate (default 96)
    FAULT_INJECT: when "1", replaces model output with "not json"
"""
import json
import logging
import os
import socket
import tempfile
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_URI = os.environ.get("MODEL_URI", "/models/v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs.data.svc.cluster.local:8333")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "96"))
FAULT_INJECT = os.environ.get("FAULT_INJECT", "0") == "1"
HOSTNAME = os.environ.get("HOSTNAME", socket.gethostname())

# Serving runs under a 1-CPU cgroup limit; letting PyTorch spawn one thread per
# host core causes severe oversubscription (and ~5x worse p95 latency), so the
# thread count is pinned and can be overridden with TORCH_NUM_THREADS.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

VALID_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query", "recommendation_events",
}
REQUIRED_KEYS = {"intent", "date", "time", "place"}

_model = None
_tokenizer = None
_ready = False
_lock = threading.Lock()


def download_from_s3(uri, local_dir):
    """Download model files from S3 bucket."""
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local-access-key"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local-secret-key"),
        region_name="us-east-1",
    )
    parts = uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    log.info("Downloading model from s3://%s/%s to %s", bucket, prefix, local_dir)
    paginator = s3.get_paginator("list_objects_v2")
    found_any = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found_any = True
            key = obj["Key"]
            rel_path = key[len(prefix):].lstrip("/")
            if not rel_path:
                continue
            dest = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)

    if not found_any:
        raise FileNotFoundError(f"No objects found in s3://{bucket}/{prefix}")
    return local_dir


def load_model():
    """Load model in background thread, setting _ready to True on completion."""
    global _model, _tokenizer, _ready

    model_path = MODEL_URI
    if MODEL_URI.startswith("s3://"):
        local_dir = tempfile.mkdtemp(prefix="model-")
        # Object storage may still be starting (or briefly unavailable) when the
        # Pod boots, so retry with backoff instead of never becoming Ready.
        for attempt in range(1, 31):
            try:
                model_path = download_from_s3(MODEL_URI, local_dir)
                break
            except Exception as exc:
                log.error("Model download attempt %d failed for %s: %s", attempt, MODEL_URI, exc)
                if attempt == 30:
                    return
                time.sleep(10)

    if not os.path.exists(model_path):
        log.error("Model path does not exist: %s", model_path)
        return

    log.info("Loading model weights from %s", model_path)
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(model_path)
        _model.eval()
        _ready = True
        log.info("Model ready (version=%s)", MODEL_VERSION)
    except Exception as exc:
        log.error("Failed to load model from %s: %s", model_path, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()
    yield


app = FastAPI(title="Command Parser Service", lifespan=lifespan)


def add_headers(response: JSONResponse) -> JSONResponse:
    response.headers["X-Served-By"] = HOSTNAME
    response.headers["X-Model-Version"] = MODEL_VERSION
    return response


def extract_json_object(text):
    """Parse the model output, tolerating leading/trailing text.

    The model occasionally emits a valid object followed by extra tokens (or a
    shortened object when it hits the token limit). We locate the first balanced
    ``{...}`` block and validate that, instead of rejecting the whole string.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def validate_output(obj):
    """Validate JSON object against schema.json."""
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != REQUIRED_KEYS:
        return False
    if obj.get("intent") not in VALID_INTENTS:
        return False
    for slot in ("date", "time", "place"):
        val = obj.get(slot)
        if val is not None and not isinstance(val, str):
            return False
    return True


@app.get("/healthz")
async def healthz():
    return add_headers(JSONResponse({"status": "ok"}))


@app.get("/readyz")
async def readyz():
    if _ready:
        return add_headers(JSONResponse({"status": "ready"}))
    return add_headers(JSONResponse({"status": "not_ready"}, status_code=503))


@app.post("/parse")
async def parse(request: Request):
    if not _ready:
        return add_headers(JSONResponse({"error": "model_not_ready"}, status_code=503))

    try:
        body = await request.json()
    except Exception:
        return add_headers(JSONResponse({"error": "invalid_json_body"}, status_code=400))

    text = body.get("text")
    if not text or not isinstance(text, str):
        return add_headers(JSONResponse({"error": "missing_or_invalid_text"}, status_code=400))

    try:
        prompt = f"Command: {text}\nJSON:"
        inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)

        # Generation is CPU-bound and blocking: run it in the worker thread so
        # the event loop stays free to answer the liveness/readiness probes.
        def _generate():
            with torch.no_grad():
                return _model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=_tokenizer.pad_token_id,
                )

        outputs = await run_in_threadpool(_generate)

        full_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = full_text[len(prompt):].strip()

        # FAULT_INJECT: replace model output with "not json" before checking
        if FAULT_INJECT:
            prediction = "not json"

        parsed = extract_json_object(prediction)
        if parsed is None:
            return add_headers(JSONResponse({"error": "invalid_model_output"}, status_code=500))

        if not validate_output(parsed):
            return add_headers(JSONResponse({"error": "invalid_model_output"}, status_code=500))

        return add_headers(JSONResponse(parsed))

    except Exception as exc:
        log.error("Inference exception: %s", exc)
        return add_headers(JSONResponse({"error": "invalid_model_output"}, status_code=500))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
