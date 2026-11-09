# Complete Guide: Training Your Cloud Assignment in Google Colab (Free T4 GPU)

## Why Did Colab Show `https://colab.research.google.com/drive/cloud_assignment_artifacts.zip`?
In Google Colab, standard HTML links (like `FileLink`) do not work because Colab rewrites relative URL links to its own web interface, causing a `404: Path doesn't exist` error.

In Google Colab, files must be downloaded using:
```python
from google.colab import files
files.download("cloud_assignment_artifacts.zip")
```

Also, a new Colab runtime starts with an empty `/content/` directory, so you must upload the small project starter files (`colab_starter.zip`, ~80 KB) before running the training script!

---

## 4 Simple Steps to Run in Google Colab

### Step 1: Open Google Colab with GPU
1. Go to [colab.research.google.com](https://colab.research.google.com) and click **"New Notebook"**.
2. Click **Runtime** $\to$ **Change runtime type** $\to$ Select **T4 GPU** $\to$ Click **Save**.

---

### Step 2: Run Cell 1 (Install Dependencies)
Copy and paste this into the first code cell and run it:
```python
!nvidia-smi
!pip install -q transformers torch accelerate datasets pyyaml boto3

import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

### Step 3: Run Cell 2 (Upload Project Files)
Copy and paste this into a second code cell:
```python
from google.colab import files
import os

print("Please select and upload 'colab_starter.zip' from your computer:")
uploaded = files.upload()

for fname in uploaded.keys():
    if fname.endswith(".zip"):
        !unzip -o "{fname}"
        print(f"Extracted {fname} successfully!")

!ls -la
```
When you run this cell, a **"Choose Files"** button will appear:
- Click **"Choose Files"**.
- Select the `colab_starter.zip` file located in your project folder:  
  `/Users/birampraveen/Downloads/cloud_assignment/colab_starter.zip` (it is only ~80 KB and uploads in 2 seconds!).

---

### Step 4: Run Cell 3 (Start the Training Pipeline)
In a new code cell, run:
```python
!python scripts/kaggle_pipeline.py
```
This runs the full pipeline on Colab's T4 GPU in **~12–15 minutes**:
1. **Teacher Labelling**: Batched GPU inference using `microsoft/Phi-3.5-mini-instruct` on 3,000 commands with greedy decoding (`temperature=0, seed=42`).
2. **Sharding**: Generates `shard-0.jsonl` through `shard-3.jsonl`.
3. **Data Cleaning**: Runs `clean.py` (rules a–e), checks the invariant, creates `train.jsonl` and `train.sha256`.
4. **Student Distillation**: Fine-tunes `EleutherAI/pythia-70m` on 1,500 records $\to$ creates `models/v1/` and `loss.csv`.
5. **Version 2 Training**: Fine-tunes `models/v2/` and `loss_v2.csv`.
6. **Multi-Model Evaluation**: Runs `tools/score.py` on `validation.jsonl` for Teacher, Base Pythia-70M, and Student v1 & v2.
7. **Release Packaging**: Archives models into `release/*.tar.gz`.
8. **Creates Output Zip**: Builds `cloud_assignment_artifacts.zip`.

---

### Step 5: Run Cell 4 (Download the Generated Artifacts)
In the final code cell, run:
```python
from google.colab import files
import os

zip_name = "cloud_assignment_artifacts.zip"
if os.path.exists(zip_name):
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"Downloading {zip_name} ({size_mb:.1f} MB)...")
    files.download(zip_name)
else:
    print(f"Error: {zip_name} does not exist. Check Cell 3 logs.")
```
Your browser will automatically download `cloud_assignment_artifacts.zip` to your computer's `Downloads` folder!

---

### Step 6: Extract into Local Project
Back on your Mac terminal:
```bash
cd /Users/birampraveen/Downloads/cloud_assignment
unzip -o ~/Downloads/cloud_assignment_artifacts.zip
```
Now all trained models (`models/v1`, `models/v2`), evaluation scores, logs, and release tarballs will be in place!

