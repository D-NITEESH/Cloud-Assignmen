# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

# Install CPU-only PyTorch from the official CPU index first (small wheel, no
# CUDA libraries), then the remaining pinned dependencies from PyPI. Keeping
# the two installs separate avoids cross-index metadata/hash conflicts.
RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0+cpu \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

# Copy the pre-built virtual environment
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy application code only — strictly no model weights baked in
COPY app.py .

EXPOSE 8000

ENV MODEL_URI=/models/v1
ENV MODEL_VERSION=v1
ENV MAX_NEW_TOKENS=64
ENV FAULT_INJECT=0

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
