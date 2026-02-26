## Python Scripts Usage

This repo focuses on estimating image token usage and costs for TotK frames.
Below is the typical execution flow and how to run each script.

### 0. Prerequisites

- Place target frame images (`.jpg`) in the `frames/` directory.
  All scripts (upload, prompt generation, cost estimation, etc.) expect this directory to exist.

### 1. Environment

- Use uv in the current directory.
- Required packages: `pandas`, `tqdm`, `openai`

Install if needed:

```
uv pip install pandas tqdm openai
```

### 2. Summarize labels

Summarizes label distributions from `frame_label.csv`.

```
uv run python 1_summarize_labels.py
```

### 3. Build balanced datasets

Creates per-scenario CSVs under `balanced_datasets/`.

```
uv run python 2_build_balanced_datasets.py
```

Outputs:
- `balanced_datasets/all_balanced.csv.gz`
- `balanced_datasets/oob_balanced.csv.gz`
- `balanced_datasets/fg_balanced.csv.gz`

### 4. Summarize totals across scenarios

Aggregates totals for multiple scenarios and writes a CSV summary.

```
uv run python 3_summarize_scenario_costs.py
```

Output:
- `prompts/costs_summary.csv`

Options:
- `--scenarios all oob fg`
- `--image-token-model gpt-4.1-mini`
- `--pricing-tier standard|batch`
- `--window 10`

### 5. Upload frames to OpenAI

Uploads all frames in `frames/` to the OpenAI Files API and saves the mapping CSV.
Requires `OPENAI_API_KEY` in `.env`.

```
uv run python 4_upload_frames.py
```

Output:
- `frame_file_ids.csv` (frame name -> file_id mapping)

Notes:
- Resumable: already-uploaded frames are skipped on re-run.
- `--sleep 0.1` controls delay between uploads to avoid rate limits.

### 6. Build per-image prompts

Creates per-image prompt JSONL in Responses API format using `file_id`.
Each prompt contains the target frame + context frames (±10).

```
uv run python 5_build_image_prompts.py --scenario all
```

Output:
- `prompts/<scenario>_vision.jsonl`

Options:
- `--scenario all|oob|fg`
- `--window 10`
- `--model gpt-4.1-mini`
- `--mapping frame_file_ids.csv`

### 7. Run vision prompts against OpenAI API

Reads prompt JSONL from `prompts/` and sends each to the Responses API.
Resumable: already-processed frames are skipped on re-run.

```
uv run python 6_run_vision_prompts.py --scenario all
```

Output:
- `results/<scenario>_vision_results.jsonl`

Options:
- `--scenario all|oob|fg`
- `--output results/custom_output.jsonl`
- `--sleep 0.1` (delay between requests)
- `--max-tokens 16`
- `--temperature 0.0`
- `--max-retries 3`

### 8. Optional utilities

- `estimate_image_costs.py`
  Prints total image tokens and estimated cost for one scenario.
  (Shared utility module — not numbered because it is imported by other scripts.)
