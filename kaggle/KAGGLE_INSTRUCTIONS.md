# Scoring on Kaggle (dual T4)

## Why

Locally this runs on CPU at ~40s per image (three autoencoder encode/decode round
trips plus LPIPS). At 8,024 images (1,024 real, 7,000 generated across 7 models)
that is ~3.7 days — not practical. Two T4s running concurrently bring it down to
roughly 1.5-2.5 hours, well inside the 12h session cap and the 30h/week quota.

## What you get

One file, `score_dual_gpu.py`, produces:

- `results/real_scores.csv` — columns `file,delta_min`
- `results/gen_scores.csv` — columns `model,file,delta_min`

The `model` column is what lets `src/evaluate.py` reproduce the paper's
**per-generator** Table 1 / Table 3 numbers. The paper never reports a pooled
real-vs-all-generators figure, so per-generator is the only fair comparison.

## Step 1 — Notebook setup

1. https://www.kaggle.com/code → **New Notebook**
2. Right panel → **Settings**:
   - Accelerator: **GPU T4 x2** (both are used; with only one visible the script still
     runs, just at half speed, and says so)
   - Internet: **On** — needed to pip install and to pull the SD1/SD2/Kandinsky
     autoencoder weights (~2-3GB, one time)
3. Right panel → **Data** → **Add Input** → select your existing dataset (the one
   already holding `real/` and `generated/`). No re-upload needed.

## Step 2 — Install dependencies

```
!pip install -q diffusers==0.39.0 transformers accelerate lpips
```

(Pinned — newer `diffusers` on the current Kaggle image spam Flax-deprecation warnings and have
caused breakage; 0.39.0 is confirmed working.)

Torch/torchvision/PIL/numpy are already on the Kaggle GPU image.

## Step 3 — Run the scorer

Paste the entire contents of `kaggle/score_dual_gpu.py` into one cell. Before running,
set `DATA_ROOT` near the top to your dataset's mount path:

```python
DATA_ROOT = "/kaggle/input/datasets/mustakimbinmohsin/image-dataset/data"
```

If unsure of the exact path, run `!find /kaggle/input -maxdepth 3` first. The script
expects `<DATA_ROOT>/real/` and `<DATA_ROOT>/generated/<model>/`; if Kaggle nested
things one level deeper (e.g. `.../aeroblade-data/data/real`), include that segment.

Run the cell. You will see both GPUs report independently:

```
[cuda:0] 4024 images (1024 real, 3000 generated) from: real + CompVis-stable-diffusion-v1-1-ViT-L-14-openai, kandinsky-community-kandinsky-2-1-ViT-L-14-openai, midjourney-v4, midjourney-v5
[cuda:1] 3000 images (0 real, 3000 generated) from: runwayml-stable-diffusion-v1-5-ViT-L-14-openai, stabilityai-stable-diffusion-2-1-base-ViT-H-14-laion2b_s32b_b79k, midjourney-v5-1
[cuda:0] 200/4024  000123.png: 0.0081  elapsed=1.4min  rate=2.38img/s  eta=32.0min
```

Progress prints every 200 images per GPU with a live rate and ETA. The cell blocks
until both processes finish, so its runtime reflects the parallel (not summed) time.

Prefer **Save & Run All (Commit)** if you would rather not babysit it — it runs
headless in the background and you collect the output afterwards.

### Why it is written this way

The script writes its worker out to a real `.py` file and launches it twice with
`subprocess`, rather than using `multiprocessing.Process`. That is deliberate:
`multiprocessing`'s `spawn` start method (required for CUDA) cannot pickle a function
defined in a notebook cell, so the multiprocessing version dies with
`AttributeError: Can't get attribute '_worker' on <module '__main__'>` when pasted
into a cell. Separate `python` invocations communicate only through files, so pasting
works.

Real and generated images are assigned to a GPU as whole units, not interleaved: GPU 0
gets all of `real/` plus whichever `generated/<model>/` directories are listed in
`GPU0_GENERATED_MODELS` at the top of the script; GPU 1 gets every other
`generated/<model>/` directory, in full. No directory is ever split across the two
GPUs. Edit `GPU0_GENERATED_MODELS` if you add or remove generator folders.

## Step 4 — Bring the results back

Download `real_scores.csv` and `gen_scores.csv` from the notebook's **Output** panel
(or **Data → Output** after a commit run) into the **project root**:

```
e:/Paper/SPL3/Module1/real_scores.csv
e:/Paper/SPL3/Module1/gen_scores.csv
```

Then locally:

```
venv312\Scripts\python src/evaluate.py
venv312\Scripts\python src/calibrate_target_fpr.py --target-fpr 0.05
```

`evaluate.py` prints per-generator AP and TPR@5%FPR side by side with the paper's
published values. `calibrate_target_fpr.py` writes `models/threshold.json`, which
`src/detect.py` picks up automatically.

## Sanity check before trusting anything

The paper's Fig. 3 shows generated images below ~0.02 and real images spread across
~0.02-0.065. If your generated scores run much higher than that, the LPIPS layer index
is wrong again — it must be **1** (0-based), which is the paper's LPIPS₂. Index 2 is
the paper's LPIPS₃ and measurably degrades separability.
