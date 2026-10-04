# Audio-Visual Anomaly Detection (GRID corpus reproduction)

Reproduction of Feng, Chen & Owens, *"Self-Supervised Video Forensics by
Audio-Visual Anomaly Detection"* (`audio + video.pdf` in this folder), trained on
the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/) instead of the paper's
LRS2/LRS3. Self-contained under `Speech/`, the same way `src/` roots the repo's other
module (AEROBLADE image forensics).

## Two pipelines: from-scratch (undeployed) vs. pretrained-SyncNet (deployed)

This folder documents two different attempts at the same detector, and only one of
them backs the desktop app's "Speech Forensics" screen:

- **From-scratch** (`train_sync.py`, `models/sync_model.py`, `extract_features.py`,
  `train_ar.py`, `calibrate.py`, `detect.py`) — the original plan: train a custom
  sync-embedding encoder from scratch. **Never completed** — no checkpoint exists
  anywhere under `models/sync/` or `models/ar/`, so this code path doesn't currently
  run. Kept for documentation; the rest of this README (below the next section)
  describes it as originally written.
- **Pretrained-SyncNet** (`syncnet_wrapper.py`, `detect_pretrained.py`,
  `models/ar_model.py`'s `load_pretrained_ar_model()`, `vendor/syncnet_python/`) —
  what actually shipped. Swaps the custom sync encoder for the official pretrained
  SyncNet (`syncnet_v2.model`) + S3FD face detector (`sfd_face.pth`), trained via
  `kaggle/speechcode.ipynb`, and trains only the autoregressive anomaly model on top.
  Reuses this file's `TAU`/`DELAY_BINS`/`FPS`/`AUDIO_SR` constants as-is — same paper
  formulation (Eq. 1/4/5), different feature extractor.

### Quickstart (pretrained-SyncNet)

```
venv312\Scripts\pip install -r Speech\requirements-av.txt -r Speech\requirements-syncnet.txt
python Speech\detect_pretrained.py --video path\to\clip.mp4 --plot
```

Prerequisites:

- **ffmpeg on PATH** — both `vendor/syncnet_python/run_pipeline.py` and
  `SyncNetInstance.evaluate()` shell out to it directly (frame/audio extraction).
- The three checkpoint files in `models/`:
  - `syncnet_v2.model` (54.6MB) and `sfd_face.pth` (89.8MB) — the official Oxford VGG
    checkpoints, re-downloadable from
    `https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/` (see
    `kaggle/speechcode.ipynb`'s Cell 0).
  - `speech_forensics_model.pt` (7MB) — this project's own Kaggle-trained artifact
    (state dict + config + a threshold validated against a desync positive-control
    test). **Not reproducible from a public URL** — if lost, retrain via
    `kaggle/speechcode.ipynb`.
- `vendor/syncnet_python/` present (see `vendor/syncnet_python/VENDORED.md` for
  provenance/license/the one patch applied).

### The threshold: why `thresholds["p95"]` must not be used

Found during integration testing, when a genuinely real, correctly-synced GRID clip
came back "Anomalous". The checkpoint's `thresholds` dict is a percentile table of
the AR model's anomaly score over real held-out video:

| key | value |
|---|---|
| p50 | 1.7982 |
| p75 | 2.0064 |
| p90 | 2.1848 |
| **p95** | **2.0733**  ← lower than p90 |
| p99 | 2.5191 |

`p95 < p90` is impossible for a real percentile sequence. The cause is in
`kaggle/speechcode.ipynb` Cell 14: it **overwrote** `thresholds["p95"]` with
`(max_real_score + min_fake_score) / 2` from a 20-clip desync test, deliberately
keeping the key name "for compatibility". That value actually sits between p75 and
p90 — around the **~80th percentile of genuine real video** — so using it flags
roughly **one in five authentic clips**. It was chosen to maximize separation on a
small, easy, seen-speaker sample, not to control false positives.

`detect_pretrained.py` therefore defaults to **p99** (`DEFAULT_THRESHOLD_KEY`), an
intact percentile that bounds false accusations at ~1% of the training distribution
while leaving the desync positive control well clear of the line. This mirrors a
decision this repo already made deliberately for the image module, where
`src/calibrate_target_fpr.py` replaced F1-maximization for the same reason
(`ARCHITECTURE.md` §5). Override with `--threshold-key` / `--threshold`;
`resolve_threshold()` structurally detects and flags a non-percentile entry rather
than hardcoding the number, so it stays correct if the model is retrained.

Measured on GRID speaker `s30` (never seen in training — the checkpoint's
`dataset_info` lists only `s1`-`s8`), with the p99 threshold of 2.5191:

| clip | SyncNet offset | conf | mean score | verdict |
|---|---|---|---|---|
| original `.mpg`, untouched | 0 | 5.46 | 2.256 | Consistent with real ✓ |
| same clip, re-encoded h264/aac | 0 | 5.58 | 2.410 | Consistent with real ✓ |
| same clip, audio shifted 1.5s | −15 | 0.82 | 2.834 | Anomalous ✓ |

### Aggregation: why the verdict is a disjunction, not a mean

Mean-over-windows alone only catches *uniformly* fake video. A clip where one
stretch is dubbed or spliced averages its bad windows against its good ones and
lands under the line — observed on a real manipulated clip whose per-window scores
ran 2.16 / 2.16 / 2.30 / 2.50 / 2.64 (mean 2.35, under a 2.52 threshold) while its
delay band visibly drifted off centre in the second half.

So `detect_pretrained.py` reports three levels:

| level | rule |
|---|---|
| `real` | neither rule fires |
| `suspicious` | a sustained local region fires, but the clip mean does not |
| `anomalous` | the clip mean exceeds the threshold |

The localized rule (`find_anomalous_regions`) is deliberately **not** a bare
max-over-windows: taking a max over N windows inflates the false-positive rate with
clip length. Instead it requires **persistence** — a run of at least
`MIN_RUN_WINDOWS` windows — which is the more robust correction when windows overlap
by `seq_len - stride` and their scores are therefore strongly correlated. Run extent
uses **hysteresis** (trigger at the main threshold, extend through neighbours above
`EXTEND_THRESHOLD_KEY`), because a window straddling a manipulated segment's
boundary genuinely contains both real and fake frames and so is diluted by
construction — the same rationale as Canny edge hysteresis. Flagged regions are
reported with frame *and* second timestamps, and shaded on both panels of the figure.

Two caveats that remain even after the threshold fix: the model saw only 8 speakers,
so scores on unfamiliar speakers do sit somewhat higher (2.26 vs. a training real
mean of 1.81), leaving a thinner margin than the numbers suggest; and re-encoding
alone costs ~0.15 of that margin. Treat verdicts as indicative, not definitive, and
read them alongside the SyncNet offset/confidence the detector now reports — those
are direct physical measurements and separated far more cleanly than the AR score in
every test above.

## What this reproduces (from-scratch pipeline, as originally planned)

1. **Synchronization model** (`models/sync_model.py`, trained by `train_sync.py`): a
   visual encoder (torchvision's MC3-18 — the "ResNet-18 2D+3D" the paper cites) +
   audio encoder (a VGG-M-style CNN on log-mel spectrograms) + a small Transformer
   fusion head, trained with InfoNCE to recognize which audio truly aligns with which
   video, in a two-phase curriculum (cross-video negatives, then within-video delay
   negatives — paper Appendix A.5).
2. **Feature extraction** (`extract_features.py`): runs the trained sync model over
   every real clip to produce a per-frame time-delay probability distribution
   S(i,·) — 31 bins for delays in [-15, +15] frames (paper Eq. 1).
3. **Autoregressive anomaly model** (`models/ar_model.py`, trained by `train_ar.py`):
   a decoder-only causal Transformer trained **only on real videos** to predict each
   frame's S(i,·) from the previous frames', via soft cross-entropy (paper Eq. 5 —
   the paper's best-performing feature/loss combination, Table 3).
4. **Detection** (`calibrate.py`, `detect.py`): a video's anomaly score is the AR
   model's cumulative prediction loss (paper Eq. 4) — high loss means its
   audio-visual timing looks unlike anything in the real training distribution.

**What's out of scope for now**: real evaluation against a manipulated-video dataset
(FakeAVCeleb/KoDF, as in the paper). No such dataset is downloaded yet, so
`make_synthetic_fakes.py` + `evaluate.py` only sanity-check the pipeline against
clips with deliberately desynced audio — useful for catching bugs, not a forensics
benchmark. Swap in a real fake-video set later without touching stages 1-3 at all.

## Advance requirements

Before running anything:

1. **ffmpeg** on PATH (Windows build — e.g. via `winget install ffmpeg` or a static
   build from ffmpeg.org). Check with `ffmpeg -version`. Needed by `preprocess.py`
   to demux GRID's `.mpg` files into audio.
2. **Extra Python packages**, on top of the repo's main `requirements.txt`:
   ```
   venv312\Scripts\pip install -r Speech\requirements-av.txt
   ```
   (opencv-python + mediapipe, for face cropping — `preprocess.py` only. Every
   training/inference script needs nothing beyond torch/torchvision/numpy/scipy,
   already installed, so they run unmodified on Kaggle.)
3. **A GRID corpus subset**, downloaded manually from
   https://spandh.dcs.shef.ac.uk/gridcorpus/ — **normal-quality video** (360x288)
   and **25kHz audio**, for however many speakers you want (9+ recommended, so the
   3-way split below has enough per group). Extract each speaker so you end up with:
   ```
   Speech/data/grid_raw/
     s1/video/s1/*.mpg
     s1/audio/s1/*.wav
     s2/video/s2/*.mpg
     s2/audio/s2/*.wav
     ...
   ```
   If your zip extracts to a different nesting, either move files to match this or
   pass `--video-glob`/`--audio-glob` to `manifest.py`.
4. **A Kaggle account** with GPU quota, for the actual training (see
   `kaggle/KAGGLE_INSTRUCTIONS.md`) — this machine is CPU-only.

## Step by step

### 1. Local: manifest + preprocess

```
venv312\Scripts\python Speech\manifest.py --speakers s1,s2,s3,s4,s5,s6,s7,s8,s9
venv312\Scripts\python Speech\preprocess.py
```

`manifest.py` splits your speakers 3 ways — `sync_train` / `ar_train` /
`heldout_real` — deterministically (speaker-disjoint, so no speaker's utterances
leak across stages) and writes `data/manifest.csv`. `preprocess.py` face-crops every
utterance to a 112x112 clip (`frames.npy`) and resamples its audio to 16kHz
(`audio.wav`) under `data/grid_processed/`; it's resumable, so reruns just top up
whatever's missing. Check the printed per-status counts (`success`/`no_face`/etc.)
— a handful of `no_face` is normal, a majority is not (check crop margin / that
the videos actually contain a visible face).

### 2. Local: smoke test

```
venv312\Scripts\python Speech\smoke_test.py
```

No data needed — builds random tensors of the right shapes and runs one
forward+backward step of both models on CPU. Confirms the architectures and loss
functions are wired correctly before spending any Kaggle GPU time.

### 3. Kaggle: train both models

Follow `kaggle/KAGGLE_INSTRUCTIONS.md` in full. Short version: upload
`Speech/` (code + `data/manifest.csv` + `data/grid_processed/`, **not**
`data/grid_raw/`) as a Kaggle Dataset, then in a GPU notebook:

```
python train_sync.py --phase1-steps 3000 --phase2-steps 6000
python extract_features.py --splits ar_train,heldout_real
python train_ar.py --steps 8000
python calibrate.py --target-fpr 0.05
```

Download `models/sync/sync_model.pt`, `models/ar/ar_model.pt`, and
`models/threshold.json` back into your local `Speech/models/`.

### 4. Local: sanity check + detect

```
venv312\Scripts\python Speech\make_synthetic_fakes.py --split heldout_real
venv312\Scripts\python Speech\evaluate.py
venv312\Scripts\python Speech\detect.py --processed-dir Speech\data\grid_processed\s5\bbaf2n --plot
```

`evaluate.py` reports AP/AUC of real vs. deliberately-desynced clips — a sanity
check, not a forensics benchmark (see above). `detect.py` scores one clip end to end
and prints a verdict against the calibrated threshold; `--video path\to\clip.mp4`
also works for a brand-new raw video (runs face-crop + audio-extract first, so needs
the same opencv/mediapipe/ffmpeg setup as preprocessing).

## Module layout

| File | Role |
|---|---|
| `config.py` | Paths + all paper hyperparameters (fps, sample rate, tau, model dims, optimizer settings) |
| `manifest.py` | Scans `data/grid_raw/`, builds `data/manifest.csv` with the 3-way speaker split |
| `preprocess.py` | Face crop + audio resample (needs opencv/mediapipe/ffmpeg) |
| `datasets.py` | `SyncDataset` (stage 1), `FeatureSequenceDataset` (stage 2) |
| `models/sync_model.py` | Visual/audio encoders + fusion head + `SyncModel` |
| `models/ar_model.py` | `ARModel` (causal Transformer) + loss/scoring helpers |
| `train_sync.py` | Stage 1 training (two-phase curriculum) |
| `extract_features.py` | Runs the trained sync model to produce S(i,·) sequences |
| `train_ar.py` | Stage 2 training (soft-CE on real sequences only) |
| `calibrate.py` | Picks the anomaly-score threshold from `heldout_real` |
| `make_synthetic_fakes.py` | Builds a desynced-audio placeholder eval set |
| `evaluate.py` | AP/AUC on real vs. synthetic-fake scores |
| `detect.py` | Single-video inference + verdict |
| `smoke_test.py` | CPU-only, no-data shape/gradient check |
| `kaggle/KAGGLE_INSTRUCTIONS.md` | How to run the two training stages on a Kaggle GPU |
