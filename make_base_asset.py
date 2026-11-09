"""Create the base (untrained) Pythia-70M release asset for CI quality gate failure demo.

Saves the untrained EleutherAI/pythia-70m model and tokenizer into release/
and packages it as release/pythia70m-base-v1.tar.gz with checksums.
"""
import hashlib
import os
import tarfile

from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    model_name = "EleutherAI/pythia-70m"
    target_dir = "models/base"
    release_dir = "release"
    tarball_path = os.path.join(release_dir, "pythia70m-base-v1.tar.gz")

    print(f"Downloading base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(release_dir, exist_ok=True)

    print(f"Saving to {target_dir}...")
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)

    # Compute checksums
    checksum_lines = []
    for root, _, files in os.walk(target_dir):
        for fname in sorted(files):
            if fname == "checksums.sha256":
                continue
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, target_dir)
            checksum_lines.append(f"{sha256_file(fpath)}  {relpath}\n")

    with open(os.path.join(target_dir, "checksums.sha256"), "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)

    print(f"Archiving to {tarball_path}...")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(target_dir, arcname="v1")

    print(f"Created {tarball_path} (SHA-256: {sha256_file(tarball_path)})")


if __name__ == "__main__":
    main()

