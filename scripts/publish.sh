#!/usr/bin/env bash
# Publish a trained model to SeaweedFS and create a release tarball.
#
# Usage:
#   bash scripts/publish.sh v1   # publish models/v1
#   bash scripts/publish.sh v2   # publish models/v2

set -euo pipefail

VERSION="${1:-v1}"
MODEL_DIR="models/${VERSION}"
RELEASE_DIR="release"
S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:8333}"
BUCKET="models"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory $MODEL_DIR does not exist"
    exit 1
fi

echo "=== Publishing model ${VERSION} ==="

# Upload to S3
echo "Uploading to s3://${BUCKET}/${VERSION}/ ..."
python3 - <<PYEOF
import boto3, os, sys
s3 = boto3.client("s3",
    endpoint_url="${S3_ENDPOINT}",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local-access-key"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local-secret-key"),
    region_name="us-east-1")

model_dir = "${MODEL_DIR}"
version = "${VERSION}"
bucket = "${BUCKET}"

for root, dirs, files in os.walk(model_dir):
    for fname in files:
        local_path = os.path.join(root, fname)
        rel_path = os.path.relpath(local_path, model_dir)
        key = f"{version}/{rel_path}"
        print(f"  {local_path} -> s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)

print("S3 upload complete.")
PYEOF

# Create release tarball with checksums
mkdir -p "$RELEASE_DIR"
TARBALL="${RELEASE_DIR}/pythia70m-command-parser-${VERSION}.tar.gz"

echo "Creating tarball: ${TARBALL}"

# Generate per-file SHA-256 checksums
cd "$MODEL_DIR"
find . -type f -exec shasum -a 256 {} \; > checksums.sha256
cd - > /dev/null

tar -czf "$TARBALL" -C "$(dirname "$MODEL_DIR")" "$(basename "$MODEL_DIR")"

echo "Tarball SHA-256:"
shasum -a 256 "$TARBALL"

echo "=== Done: ${VERSION} published ==="
