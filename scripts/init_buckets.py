"""Create S3 buckets and upload the command pool to SeaweedFS.

Usage:
    python scripts/init_buckets.py --endpoint http://localhost:8333

Idempotent: safe to run multiple times.
"""
import argparse
import os

import boto3
from botocore.exceptions import ClientError


def get_s3(endpoint):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "local-access-key"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "local-secret-key"),
        region_name="us-east-1",
    )


def ensure_bucket(s3, name):
    """Create a bucket if it does not exist."""
    try:
        s3.head_bucket(Bucket=name)
        print(f"  bucket '{name}' already exists")
    except ClientError:
        s3.create_bucket(Bucket=name)
        print(f"  bucket '{name}' created")


def upload_pool(s3, bucket="datasets", key="commands_pool.jsonl",
                local="data/commands_pool.jsonl"):
    """Upload the command pool file."""
    if not os.path.exists(local):
        print(f"  WARNING: {local} not found, skipping upload")
        return
    s3.upload_file(local, bucket, key)
    print(f"  uploaded {local} -> s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Initialize S3 buckets")
    parser.add_argument("--endpoint", default="http://localhost:8333",
                        help="S3 endpoint URL")
    args = parser.parse_args()

    s3 = get_s3(args.endpoint)

    print("Creating buckets:")
    ensure_bucket(s3, "datasets")
    ensure_bucket(s3, "models")

    print("Uploading command pool:")
    upload_pool(s3)

    print("Done.")


if __name__ == "__main__":
    main()
