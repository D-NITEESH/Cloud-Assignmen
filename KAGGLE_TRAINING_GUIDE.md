# Complete Guide: Training Your Cloud Assignment on Kaggle (Free GPU)

## Why Train on Kaggle?
- **On Laptop CPU:** Labelling 3,000 commands with `phi3.5` takes **4 to 12 hours**, plus another 45–60 minutes for fine-tuning.
- **On Kaggle GPU (T4 / P100):** Labelling 3,000 commands takes only **~10–12 minutes**, and fine-tuning `EleutherAI/pythia-70m` takes only **~2 minutes**!
- You get all required outputs: 4 shard files, `train.jsonl`, `train.sha256`, `models/v1/`, `models/v2/`, `loss.csv`, validation reports (`teacher_validation.json`, `student_v1_validation.json`), and release tarballs (`release/*.tar.gz`).

---

## Step-by-Step Instructions

### Step 1: Open Kaggle and Create a New Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and sign in.
2. Click **"+ Create"** $\to$ **"New Notebook"**.
3. In the right-hand panel under **Notebook options**:
   - **Accelerator**: Select **GPU T4 x2** (or GPU P100).
   - **Internet**: Ensure **Internet on** is toggled ON.

---

### Step 2: Upload Your Code to Kaggle

You have two easy options:

#### Option A: Import `kaggle_train_notebook.ipynb` directly (Recommended)
1. In the Kaggle notebook menu, click **File $\to$ Import Notebook**.
2. Upload `kaggle_train_notebook.ipynb` from your local folder:
   `/Users/birampraveen/Downloads/cloud_assignment/kaggle_train_notebook.ipynb`
3. Upload your assignment folder as a zip or clone your GitHub repository:
   ```bash
   !git clone https://github.com/<your-username>/cloud-assignment.git .
   ```

#### Option B: Zip and Upload as a Kaggle Dataset
1. Zip your local project directory (excluding any heavy virtualenvs):
   ```bash
   zip -r cloud_assignment.zip . -x ".venv/*" "__pycache__/*" ".git/*"
   ```
2. On Kaggle, click **File $\to$ Upload data** and upload `cloud_assignment.zip`.
3. In your notebook cell:
   ```bash
   !unzip -q /kaggle/input/cloud-assignment/cloud_assignment.zip -d /kaggle/working/
   %cd /kaggle/working/
   ```

---

### Step 3: Run the Complete Training Pipeline
In a code cell, run:
```bash
!pip install -q transformers torch accelerate pyyaml boto3
!python scripts/kaggle_pipeline.py
```

### What `scripts/kaggle_pipeline.py` Does Automatically:
1. **Teacher Labelling**: Batched GPU inference using `microsoft/Phi-3.5-mini-instruct` (identical weights to phi3.5) with greedy decoding (`do_sample=False, temperature=0, seed=42`).
2. **Sharding**: Writes the 4 required shard objects: `results/shards/shard-0.jsonl` through `shard-3.jsonl`.
3. **Data Cleaning**: Runs `clean.py` logic enforcing rules (a)–(e), verifies the counting invariant, generates `datasets/v1/train.jsonl` and computes `datasets/v1/train.sha256`.
4. **Student Distillation**: Fine-tunes `EleutherAI/pythia-70m` on the first 1,500 cleaned records in the exact format:
   ```
   Command: <the command>
   JSON: <the teacher’s JSON reply>
   ```
   Exports `models/v1/` and `training/loss.csv`.
5. **Version 2 Training**: Trains version 2 of the student and exports `models/v2/` and `training/loss_v2.csv`.
6. **Multi-Model Evaluation**: Runs `tools/score.py` on `data/validation.jsonl` for Teacher, Untrained Base, and Student v1 & v2.
7. **Release Packaging**: Creates tarballs with checksums:
   - `release/pythia70m-command-parser-v1.tar.gz`
   - `release/pythia70m-command-parser-v2.tar.gz`
   - `release/pythia70m-base-v1.tar.gz`
8. **Generates Download Zip**: Packages all results into `cloud_assignment_artifacts.zip`.

---

### Step 4: Download Artifacts to Your Machine
In the last cell of your Kaggle notebook, run:
```python
from IPython.display import FileLink
FileLink("cloud_assignment_artifacts.zip")
```
Click the link to download `cloud_assignment_artifacts.zip` to your computer.

---

### Step 5: Extract Artifacts into Your Project Directory
On your local laptop terminal:
```bash
cd /Users/birampraveen/Downloads/cloud_assignment
unzip -o ~/Downloads/cloud_assignment_artifacts.zip
```

Once extracted, your local workspace will instantly have:
- `datasets/v1/train.jsonl` & `train.sha256`
- `models/v1/` & `models/v2/` (ready for SeaweedFS & Docker)
- `release/*.tar.gz` (ready for GitHub Releases)
- `results/` containing `clean_counts.json`, `loss_v1.csv`, `loss_v2.csv`, `teacher_validation.json`, `student_v1_validation.json`, etc.

Now you can proceed immediately with **Phase 4 & Phase 5 (local container and Kubernetes deployment)** without waiting hours for CPU training!

