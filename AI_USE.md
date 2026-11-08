# AI Use Disclosure

As required by Task 6 of the CS3081 Cloud Computing Assignment, this document discloses all AI tools used throughout the implementation, their specific purposes, and the verification methodologies applied to ensure correctness, reproducibility, and compliance with the assignment specification.

---

## 1. AI Tools Used

| Tool | Model / Engine | Purpose |
|---|---|---|
| **Ollama** | `phi3.5:3.8b-mini-instruct-q4_K_M` | Teacher model for generating sequence-level distillation labels across the 3,000 command dataset. |
| **Antigravity Coding Assistant** | Claude Opus 4.6 / Gemini 3.8 Flash | Assisting with architecture design, Kubernetes manifest drafting, data processing scripts, unit test suites, and documentation. |

---

## 2. Details of AI Assistance and Tasks

### Task 1: Teacher Labelling and Knowledge Distillation
- **Prompt Formulation:** AI was used to draft the strict formatting prompt for the `phi3.5` teacher model in `prompts/teacher_prompt.txt`, ensuring adherence to the 8 intents and exact slot extraction rules without hallucination.
- **Data Generation:** `phi3.5` acted as the teacher model under greedy decoding (`temperature=0, seed=42`) to produce JSON labels for all 3,000 commands in `data/commands_pool.jsonl`.

### Task 2 & 3: Training & Service Containerisation
- AI was used to structure the causal LM fine-tuning script (`training/train.py`) and FastAPI microservice (`service/app.py`), specifically implementing proper readiness/liveness probe lifecycles, S3 model retrieval, and `FAULT_INJECT` validation hooks.

### Task 4 & 5: Kubernetes & Networking
- AI assisted in calculating CIDR boundaries and generating the subnet allocation table for `10.20.0.0/16` and formatting the stateless NACL rules for `network/nacl.json`.

### Task 6: CI/CD Pipeline
- AI assisted in drafting the GitHub Actions YAML configuration (`.github/workflows/ci.yml`), including the quality gate verification and Gitleaks scanning steps.

---

## 3. Verification Methodologies

All AI-generated code and data underwent rigorous automated and manual validation:

1. **Automated Unit Testing:**
   - 24 comprehensive unit tests were authored in `tests/` covering NACL evaluation, prediction scoring, schema conformity, data cleaning invariants, and service validation logic. All 24 tests execute and pass cleanly.

2. **NACL Simulation Verification:**
   - The stateless NACL rules in `network/nacl.json` were strictly verified against the sample packet trace using `tools/nacl_sim.py --check`, achieving 6/6 (100.0%) passing rate.

3. **Data Cleaning & Schema Validation:**
   - All teacher responses were programmatically validated against `data/schema.json` via `worker/clean.py`, checking for valid JSON, exact 4 keys, allowed intent enum, and verifying that every non-null slot string literally exists within the input command text. The counting invariant `input = (a) + (b) + (c) + (d) + duplicates + kept` was enforced by assertions.

4. **Model Performance Evaluation:**
   - All model outputs were evaluated using `tools/score.py` against the gold validation set `data/validation.jsonl`, measuring JSON validity, slot accuracy, and full-frame accuracy.


---

## 4. Practical Session Log (execution on this machine)

The repository was executed end-to-end on the author's macOS machine (8 GB RAM,
Apple silicon) rather than only written. AI assistance during execution, with
the verification performed for each change:

| Issue found while running | Change made | How it was verified |
|---|---|---|
| `phi3.5` continued generating invented follow-up examples after the first JSON object, exceeding the 120 s worker timeout | Added `format: "json"` and a `"\n\n"` stop sequence to the Ollama request in `worker/worker.py` and `training/evaluate.py` (temperature 0 and seed 42 unchanged) | A single call dropped from a timeout to ~2 s and returned exactly one object |
| Student evaluation used the prompt `Parse: <text>` while training used `Command: <text>\nJSON:` | Evaluation prompt aligned with `training/data_helpers.format_for_training` | Generated output inspected and parsed by `tools/score.py` |
| Training computed loss over the prompt tokens and padded every sequence to 128 tokens | Labels masked with `-100` before `JSON:`; dynamic per-batch padding | `make train SUBSET=50` ran to completion, `training/loss_subset.csv` written |
| All 2,860 cleaned records were used | `n_train: 1500` in `training/config.yaml`, first 1,500 preserved in id order | Training log prints the number of records used |
| SeaweedFS pod never became Ready | Readiness probe path corrected to `/cluster/status` on port 9333 | `kubectl -n data rollout status deploy/seaweedfs` completed |
| Container image failed to resolve `torch==2.2.1+cpu` on arm64 and hit pip index conflicts | CPU wheel pinned to `torch==2.6.0+cpu`, torch installed in its own layer, `numpy==1.26.4` pinned | Image built and `docker image inspect` reported its size |
| PyTorch used all host cores under a 1-CPU limit | `torch.set_num_threads(TORCH_NUM_THREADS, default 1)` in `service/app.py` | Latency measured with the 50-request test |

Every command above was run and its output read; the numbers quoted in the
report come from the files in `results/`, not from AI suggestions.
