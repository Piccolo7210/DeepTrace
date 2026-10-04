# Video Forensics Module — NSG video detector

Detects AI-generated video from the **Normalized Spatiotemporal Gradient (NSG)** of a
pre-trained diffusion model's score field. Training and feature extraction were done on
Kaggle (see `NSGVD_KAGGLE_HANDOFF.md`); this machine only runs **inference, on the CPU**.

What is built, verified, and still open: `STATUS.md`.

```powershell
# one video
venv312\Scripts\python Video\detect_nsgvd.py my_clip.mp4 --plot

# the acceptance check: does this PC reproduce Kaggle's features?
venv312\Scripts\python Video\detect_nsgvd.py --parity

# in the app: Home -> "Video Forensics"
venv312\Scripts\python -m app
```

## What it does

```
video -> 8 frames sampled from the first 95 (stride = 95//8 = 11)
      -> bottom 10% cropped, centre square, 224x224 LANCZOS
      -> guided-diffusion UNet (552 M params, frozen) -> score s(x_t)
      -> NSG denominator <s(x_t), dx_t> (central difference)
      -> 62 per-clip statistics -> gradient-boosting classifier
      -> p_fake, flagged if p_fake > 0.6915
```

The threshold 0.6915 was calibrated on Kaggle for a 5% false-positive rate on validation
real videos, and held at 4.8% on unseen test reals.

**Honest performance** (from the Kaggle evaluation, handoff §3):

| | AUROC | Recall at threshold |
|---|---|---|
| Validation (generators seen in training) | 0.969 | — |
| Test (unseen generators) | 0.838 | 40.9% |
| — Moonvalley I2V (seen family, unseen modality) | 0.912 | 68.6% |
| — Sora | 0.878 | 58.3% |
| — Hotshot (short, low-res) | 0.768 | 14.2% |

So a **flagged** clip is the informative outcome; "consistent with real" is weak evidence.
Known shortcut risk: in the training data, file metadata alone separated the classes
perfectly (reals were ≤568×320 at ~30 fps, fakes mostly ≥512 px at 8–24 fps), so part of
the accuracy likely comes from resolution and compression rather than from physics. These
caveats are carried in every result and in the CLI's JSON output (`Video/config.py: CAVEATS`); the app no longer displays them.

## Layout

| Path | What it is |
|---|---|
| `nsgvd_local/` | **the Kaggle bundle, unmodified** — `nsgvd/` (frame sampling, score, NSG, the 62 statistics), `third_party/guided-diffusion` @ `22e0df8`, and its own `detect.py` |
| `detect_nsgvd.py` | app-facing wrapper: `load_detector()`, `detect_video()`, `plot_report()`, CLI |
| `predictor.py` | sklearn-free evaluation of the trained trees (see below) |
| `config.py` | module constants; the "inference contract" block is not tunable |
| `models/nsg_gbm_trees.npz` | the classifier, re-encoded for numpy (0.4 MB, regenerable) |
| `tools/export_tree_dump.py` | produces that file from the `.joblib` and verifies it |
| `tests/test_video_module.py` | 16 fast checks + parity + end-to-end |
| `nsgvd_local/models/` | downloaded weights (git-ignored): UNet 2.1 GB, `.joblib`, parity file |

Everything that defines a feature value lives in `nsgvd_local/` and is used as shipped —
handoff §10 is explicit that changing the sampling, the score, the denominator or the
statistics invalidates the trained model.

## Why the classifier is not loaded with scikit-learn

The trained detector arrives as a pickled `HistGradientBoostingClassifier` that only
unpickles under **scikit-learn 1.6.1** (1.9.0 fails with `No module named '_loss'`). The
image and speech modules were built against 1.9.0, so pinning the whole project to 1.6.1
would let this module dictate versions for the other two.

Instead, `tools/export_tree_dump.py` reads the pickle once (under 1.6.1) and writes the
578 trees to a `.npz`; `predictor.py` evaluates them with numpy. This is a re-encoding,
not a retrain — same thresholds, same leaf values, same decision function — and it was
verified against scikit-learn's own `predict_proba` on 2 207 rows (random, all-zero,
all-NaN, and the 5 real parity clips): **max |dp| = 1.1e-16**, i.e. float rounding.

The project therefore still runs scikit-learn 1.9.0, and this module needs no sklearn at
all. To regenerate the dump (only needed if the model is retrained on Kaggle):

```powershell
venv312\Scripts\python -m pip install scikit-learn==1.6.1
venv312\Scripts\python Video\tools\export_tree_dump.py --verify
venv312\Scripts\python -m pip install scikit-learn==1.9.0   # restore
```

## Getting the model files

Not in git (2.1 GB of weights). With rclone configured for the `gdrive` remote:

```powershell
$RC = "D:\rclone-v1.75.1-windows-amd64\rclone-v1.75.1-windows-amd64\rclone.exe"
$dst = "Video\nsgvd_local\models"
& $RC copy gdrive:SPL3/Video-NSGVD/models/256x256_diffusion_uncond.pt $dst -P
& $RC copy gdrive:SPL3/Video-NSGVD/export/nsg_stats_gbm.joblib        $dst -P
& $RC copy gdrive:SPL3/Video-NSGVD/export/parity_samples.npz          $dst -P
(Get-FileHash "$dst\256x256_diffusion_uncond.pt" -Algorithm SHA256).Hash.ToLower()
# must start with a37c32fffd31
```

The UNet is the public OpenAI checkpoint, so it can equally be fetched from
`https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt`
(curl resumes with `-C -`; the Drive copy does not).

`detect_nsgvd.py` checks the file's exact size before loading it, because a
half-downloaded checkpoint otherwise surfaces as "checkpoint file is corrupted".

## Behaviour in the app

`app/video_engine.py` loads the detector once on a background thread at startup
(~20–40 s, ~2.2 GB RAM) and runs each analysis on a worker thread, **one video at a
time** — two concurrent UNet passes would thrash a 16 GB machine, so a second request is
rejected with a clear message rather than queued. `app/ui/video_view.py` follows
`speech_view.py`'s state machine (instructions → selected → running → results → error).

Verdicts use three levels, like the speech module: **AI-generated**, **Uncertain**
(within 0.10 of the threshold — a call that could go either way is reported as
inconclusive rather than rounded), and **Consistent with real**.

Only the first 95 frames are analysed, so for a long video the UI states how much of it
was actually judged, and suggests trimming when a later moment matters.

## Tests

```powershell
venv312\Scripts\python Video\tests\test_video_module.py
```

Groups 1–3 (classifier, stored-feature regression, frame sampling) need no weights.
Group 4 verifies the UNet hash, runs the handoff's parity check (local fp32 CPU vs
Kaggle's fp16 GPU features, tolerance 0.05) and an end-to-end analysis of a synthetic
clip; it skips with a clear reason when the weights are absent or incomplete.

## Licensing and citation

- Method: Zhang et al., *Physics-Driven Spatiotemporal Modeling for AI-Generated Video
  Detection*, NeurIPS 2025 (arXiv:2510.08073); official code `ZSHsh98/NSG-VD`.
- Score network: `openai/guided-diffusion` @ `22e0df8` (MIT) — vendored under
  `nsgvd_local/third_party/`.
- Training data: AIGVDet / GVD (Bai, Lin, Cao, Lou, PRCV 2024) plus Kinetics-400 and
  MSR-VTT reals. **The AIGVDet dataset is for academic research only, no commercial use.**
- The detector is NSG features plus a supervised gradient-boosting head — the paper's own
  NSG-BCE variant idea, not its deep-kernel MMD method, which was reproduced faithfully
  but did not learn on this data (val AUROC 0.554; handoff §2.5). Describe it accordingly.
