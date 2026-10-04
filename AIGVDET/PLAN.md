# AIGVDet — Spatio-Temporal AI-Generated Video Detection · Implementation Plan

Paper: `AIGVDET/Detection Via Spatio-Temporal Anomaly Learning.pdf` — Bai, Lin, Cao, *AI-Generated Video
Detection via Spatial-Temporal Anomaly Learning*, PRCV 2024 (arXiv 2403.16638)
Official code: https://github.com/multimediaFor/AIGVDet · Training data: Hugging Face `xmulz/AIGVDet`
Test videos: the authors' Google Drive folder (already in your Drive under `SPL3/Video-NSGVD/`)

**Status: plan only — nothing is implemented yet.** Planned 3 Oct 2026. Items marked *verified* were checked
against the real repo, HF listing, Drive and this PC while writing the plan; items marked *estimate* are
guesses that a named step must measure before anything depends on them.

**Decided by you (3 Oct):** (1) clicking **Video Forensics** opens a chooser with two options — *NSG-based* and
*AIGVDet-based*; (2) test reals come from the **existing pools only** (no GOT-10k, no clips of your own).

**At a glance**

- **What:** two ImageNet-pretrained ResNet-50 classifiers, one on RGB frames and one on RAFT optical-flow colour
  maps, fused 50/50 per frame and averaged over the clip (§2). Trained on Kaggle from the HF data, tested on the
  authors' Drive videos, run on your CPU from the app.
- **Training data (verified):** 500 + 500 train and 50 + 50 val videos as flat JPEG folders — 95 RGB frames and
  93 flow maps per video (§3.1). Not what the repo README describes.
- **Test data (verified):** 11 068 fake videos in the authors' Drive folder and **no real ones**; reals come from
  HF val + Kinetics-400 + MSR-VTT on your Drive (§3.2–3.3). The clips are short (Hotshot 8 frames, VideoCraft 16,
  NeverEnds 30 — only Moonvalley T2V has 95), span 8–30 fps, and every Kinetics/MSR-VTT real is under 448 px.
- **Biggest technical risk (measured):** the stored flow maps **cannot be regenerated** from the stored frames with
  any RAFT variant, so a flow branch trained on them may not accept maps made from a video. A cheap gate (K2) decides
  between training on the stored maps (Plan A) and on maps we regenerate (Plan B) *before* the long training (§3.4, D15).
- **CPU cost (measured):** optical flow, not the CNNs, dominates — 8–17 s per native-resolution pair on this PC,
  ~3–5 s at half scale; ResNet-50 is 0.38 s per crop. Hence three modes, chosen by a Kaggle accuracy-vs-speed
  ablation (§9).
- **Order of work:** K0 setup → K1 data/audit/benchmark → **K2 flow-recipe gate** → K3 training (RGB branch can start
  right after K1; K3b regenerates flows under Plan B) → K4 test extraction → K5 evaluation → K6 export → local
  inference → app chooser (§7–§13).

---

## 1. Context

The project ships three forensic modules: image (AEROBLADE, `src/`), speech (`Speech/`), and video
(NSG-VD, `Video/`). The NSG-VD video detector is finished but modest (test AUROC 0.84, 41 % recall at its
threshold; `Video/STATUS.md`). AIGVDet is a **second, independent video detector**: a supervised two-branch CNN
that looks at RGB frames and at optical-flow maps. NSG-VD stays untouched.

| | |
|---|---|
| Local PC (*verified*) | i5-1335U (10 cores / 12 threads), 15.7 GB RAM, Intel Iris Xe only (**no CUDA**), 16.9 GB free on `E:`; `venv312` has torch 2.13.0+cpu, torchvision 0.28.0, opencv 4.10, numpy 2.5.1 — full configuration in §1.1 |
| Kaggle | 2× T4 (16 GB each), ~30 GPU-h/week, 12 h/session, `/kaggle/working` capped at 20 GB (figures from `Video/PLAN.md` §1 — re-checked in K0) |
| Google Drive | `rclone` remote `gdrive:` works from this PC (*verified*: lists `SPL3/Video-NSGVD/`); Kaggle uses the `RCLONE_CONF` secret you already set up for NSG-VD |
| App | tkinter; one engine class per module, worker threads, `isolated_import` shim in `app/paths.py` |

**The deliverable is the demo** — a clip goes into the app, a verdict comes out — not the benchmark table.
Where "reproduce the paper exactly" and "behave sensibly on a clip unlike the training data" conflict, this plan
says so and measures both.

### 1.1 Existing environment configuration (verified 3 Oct 2026)

Read from the live machine, the Drive and the NSG handoff on the date above. This plan adds **no new local
dependency**; the versions below are the baseline every later step is written against.

**Machine and OS**

| | |
|---|---|
| OS | Windows 11 Home Single Language 10.0.26200 (25H2), 64-bit |
| CPU | 13th-gen Intel Core i5-1335U — 10 cores / 12 threads |
| RAM | 15.7 GB |
| GPU | Intel Iris Xe Graphics (driver 31.0.101.4314) — no NVIDIA/CUDA, so PyTorch runs on the CPU |
| Disk | project on `E:` — 16.9 GB free, 87.7 GB used |
| Shells | Windows PowerShell 5.1.26100.9549 (Desktop) and Git Bash. `tools/*.ps1` must be 5.1-compatible (no `&&`, `\|\|` or ternary) |

**Python environment — `E:\Paper\SPL3\SPL3Full\venv312`**

| | |
|---|---|
| Interpreter | CPython 3.12.6 (MSC v.1940, 64-bit); base `C:\Users\User\AppData\Local\Programs\Python\Python312`; `include-system-site-packages = false` |
| Always call | `venv312\Scripts\python …` — a bare `python` on PATH is the **system Python 3.13.1**, not the venv |
| `-m`, not launchers | the venv was created under `E:\Paper\SPL3\Module1\`, a folder that no longer exists (the project is now `SPL3Full`). Its console launchers (`pip.exe`, `hf.exe`, `torchrun.exe`, …) still embed that dead path and fail (checked: `pip.exe --version` exits 1 with no output). Use `venv312\Scripts\python -m pip` / `-m <tool>` |

**Libraries this plan relies on (installed versions, from `pip list`)**

| Package | Version | Role in this plan |
|---|---|---|
| torch | 2.13.0+cpu | CPU-only wheel (CUDA none); MKL and oneDNN available; 10 intra-op threads by default (the §3.4 timings used 12). ResNet-50 inference, RAFT |
| torchvision | 0.28.0+cpu | `resnet50`, `raft_large` + `Raft_Large_Weights` (import checked), `transforms` |
| numpy / scipy | 2.5.1 / 1.18.0 | arrays; `gaussian_filter` (blur-augmentation parity tests) |
| opencv-python | 4.10.0.84 | video decode and JPEG. FFmpeg backend **YES** (avcodec 58.134.100, avformat 58.76.100; Media Foundation and DirectShow yes, GStreamer no). Decoded every H.264 MP4 tried today: the authors' I2V clips, one to three clips from each T2V set, Kinetics and MSR-VTT |
| pillow | 12.3.0 | JPEG/PNG I/O, thumbnails |
| matplotlib | 3.11.1 | `plot_report` (the app forces the Agg backend) |
| pandas / scikit-learn | 3.0.5 / 1.9.0 | tables and metrics (K5 runs on Kaggle; local use optional) |
| huggingface_hub (+ hf-xet) | 1.25.1 (1.5.2) | HF download runs on Kaggle (K1); locally only for remote probing |
| tkinter | Tk / Tcl 8.6 | the app UI |
| imageio / tqdm / psutil / joblib | 2.37.4 / 4.70.0 / 7.2.2 / 1.5.3 | utilities |

Installed but not used by this plan: diffusers 0.39.0, transformers 5.14.1, accelerate 1.14.0, lpips 0.1.4,
safetensors 0.8.0, tokenizers 0.22.2, scikit-image 0.26.0, scenedetect 0.6.7.1, python_speech_features 0.6,
pyarrow 18.1.0 (71 packages in all: `venv312\Scripts\python -m pip list`).

**Health and drift**

- `pip check` reports one conflict: `huggingface-hub 1.25.1` requires `click ≥ 8.4.2` but the venv has 8.2.1.
  Harmless for this plan — HF downloads run on Kaggle and `HfFileSystem` worked in the planning probes — but the
  `hf` CLI is not relied on locally.
- The root `requirements.txt` has drifted from the venv. It lacks `opencv-python` (4.10.0.84 — a hard dependency of
  the app and of this plan), `scenedetect`, `python_speech_features` and `platformdirs`, and pins different versions
  of `click` (8.4.2 vs 8.2.1 installed), `filelock` (3.29.0 vs 3.32.0), `fsspec` (2026.4.0 vs 2026.6.0),
  `huggingface_hub` (1.24.0 vs 1.25.1), `setuptools` (78.1.0 vs 83.0.0), `tqdm` (4.69.1 vs 4.70.0) and
  `typing_extensions` (4.15.0 vs 4.16.0). The installed versions are authoritative; refreshing `requirements.txt`
  is left to M7 and is not done here.

**Tools**: git 2.46.0.windows.1 · rclone v1.75.1 (`D:\rclone-v1.75.1-windows-amd64\rclone-v1.75.1-windows-amd64\rclone.exe`,
remote `gdrive:`) · ffmpeg / ffprobe 9.0.1 (WinGet `Gyan.FFmpeg` full build, on PATH) — handy for clip audits but
**not required**: OpenCV's bundled FFmpeg decodes everything the plan needs.

**Kaggle** (recorded in `Video/NSGVD_KAGGLE_HANDOFF.md` §2.1 for the 1–2 Oct 2026 run; **not re-verified today**):
Python 3.12.13 · torch 2.10.0+cu128 · numpy 2.0.2 · pandas 2.3.3 · scikit-learn 1.6.1 · opencv 4.13.0 ·
Pillow 11.3.0 · matplotlib 3.10.0 · rclone v1.74.4 · 2× Tesla T4 · NCCL. torchvision, scipy and huggingface_hub
were not recorded, so **K0 prints them** along with every version above.

**Version gaps the plan accounts for**

- Checkpoints leave Kaggle as plain `state_dict`s (no optimizer, no pickled objects), so
  `torch.load(…, weights_only=True)` works on the local torch 2.13; `last.pt` (with optimizer state) stays on
  Kaggle and Drive.
- Different torch / numpy / OpenCV builds give tiny numerical differences → the G5 tolerance (\|Δp\| ≤ 0.03, same
  verdict), and **one JPEG implementation (OpenCV) on both sides** for the conditioning variants.
- `raft_large` needs torchvision ≥ 0.12; local is verified, and K0 asserts the import on Kaggle.

---

## 2. What the paper and the official code actually do (*verified* by reading both)

```
video ─► frames I_1..I_N (N = 95)
          ├──────────────────────► crop 448 ─► ResNet-50 (RGB branch)  ─► P_I^i ─┐
          └─► RAFT(I_i, I_i+1) ─► colour-wheel map F_i ─► crop 448 ─► ResNet-50 (flow branch) ─► P_F^i ─┤
                                                                                      P^i = 0.5·P_I^i + 0.5·P_F^i
                                                          video score P = mean_i P^i   →  fake if P > τ
```

| Item | Paper | Official code | This plan |
|---|---|---|---|
| Backbone | ResNet-50 (ImageNet), GAP → FC → sigmoid; two branches trained **separately** | `networks/resnet.py`, `fc: 2048→1` init N(0, 0.02) | torchvision `resnet50(IMAGENET1K_V1)` (same `resnet50-19c8e357.pth`); state-dict keys are identical, so the authors' released checkpoints load directly |
| Spatial input | RGB frame, **random** 448 crop (train) / **centre** 448 crop (test) | `RandomCrop((448,448))` / `CenterCrop((448,448))`, ImageNet mean/std | same |
| Temporal input | RAFT flow between adjacent frames as a 3-channel map, same crop | `demo.py`: `raft-things.pth`, `iters=20`, `InputPadder` (pad to ×8, replicate), `flow_viz.flow_to_image` (**normalised by the frame's max radius**), `cv2.imwrite` | torchvision `raft_large(C_T_V1)` = the ported `raft-things`, 20 updates. ⚠ the *stored* HF maps could not be reproduced from the stored frames (§3.4) → K2 gate, Plan A/B (D15) |
| Fusion | `P^i = α·P_I^i + (1−α)·P_F^i`, α = 0.5; `P` = mean over N−1 frames | mean RGB prob and mean flow prob, then 0.5 / 0.5 | same |
| Threshold | "video-level threshold 0.1" (§III-A) | default `-t 0.5` (`test.py`, `demo.py`) | the paper and the code **disagree** → report both, plus our own calibrated τ (§8.4) |
| Training | BCE, Adam 1e-4, ÷10 if val ACC doesn't improve for 5 epochs, stop at 1e-6; aug on 10 % of samples (blur σ0.5, JPEG 75, flip) | `trainer.py`: `BCEWithLogitsLoss`, `Adam(lr=1e-4, betas=(0.9,0.999))`, `adjust_learning_rate` ÷10 (floor 1e-6); `EarlyStopping(patience=5, delta=-0.001)` and `-0.002` after a drop; `datasets.py`: blur p=0.1 σ0.5, JPEG p=0.1 cv2 Q75, **flip p=0.5**; batch 64; frames are independent samples (`ImageFolder`); val = frame-level ACC@0.5 every epoch | same (flip p=0.5 as in code); own best-checkpoint tracker; AMP (§4) |
| Train data | 550 Moonvalley T2V + 550 YouTube-VOS reals, 10:1 train/val; 95 RGB frames + 94 flows per video | — | HF zips: 500 + 500 train videos, 50 + 50 val; 95 RGB + **93** flows per video |
| Test data | every other GVD video; reals from GOT-10k | per video: average per-frame probabilities | authors' Drive videos; reals from the existing pools (§3.3) |
| Paper targets | Table III: T2V avg **ACC 91.1 / AUC 96.9**, I2V avg **89.7 / 96.1**, Moonvalley 95.1/100, Sora 82.3/92.1; Table II: RGB-only 86.5/94.9, flow-only 89.0/95.0, fused 91.4/97.1 | — | reproduction *targets*, not promises — our real pool differs from theirs |

---

## 3. Data (*verified* while planning)

### 3.1 Training data — HF `xmulz/AIGVDet`, revision `c1326befed8ea265e547984e1e25fe83e6b2a7bf` (13.2 GB, **academic use only**)

| File | Size | JPEGs | Content |
|---|---|---|---|
| `train/moonvalley_vos2_original_crop.zip` | 7.09 GB | 95 000 = 47 500 real + 47 500 fake | RGB frames |
| `train/moonvalley_vos2_crop_op.zip` | 4.88 GB | 93 000 = 46 500 + 46 500 | RAFT flow colour maps |
| `val/moonvalley_vos2_original_crop.zip` | 0.76 GB | 9 500 = 4 750 + 4 750 | RGB frames |
| `val/moonvalley_vos2_crop_op.zip` | 0.48 GB | 9 300 = 4 650 + 4 650 | flow maps |

What the zips really contain (the README's `video_00000/00000.png` tree is **wrong** for this release, and so is
the older assumption in `Video/PLAN.md` §4.1 that these are PNGs):

- **Flat JPEG folders**, no per-video directories: `<zip-stem>/0_real/005a527edd_00000.jpg`,
  `<zip-stem>/1_fake/AAgaPRC_output_0.jpg`. Real names are `<10-hex>_<nnnnn>`, fake names `<7 letters>_output_<n>`.
  `video_id = stem.rsplit("_", 1)[0]`; the class folder gives the label (`0_real` → 0, `1_fake` → 1).
- Per video: RGB frame indices 0–94 (**95**), flow indices 0–92 (**93**; by the paper's definition flow *j* =
  RAFT(frame *j*, frame *j+1*) — but see the parity finding in §3.4). → 500 + 500 training videos, 50 + 50
  validation videos.
- **JPEG everywhere.** RGB is **1184×636 for both classes** (inferred: the YouTube-VOS clips were resized to
  Moonvalley's 1184×672 and a bottom band — the watermark crop — removed for both classes). Flow maps are
  **1184×640**. Mean file size: RGB 77 KB real / 73 KB fake; flow 53 KB / 53 KB — no obvious size shortcut.
  Min side 636 px, so `RandomCrop(448)` always fits.
- Fake IDs are Moonvalley Discord names (`XXXXXXX_output`); the Drive test clips are the *other* Moonvalley
  videos (G0 checks there is no overlap).

### 3.2 Test videos — authors' Drive folder (in your Drive; unpacked copies in `videos_subset/`)

Counts = paper Table I minus the 550 Moonvalley T2V clips used for training (same totals as
`Video/NSGVD_KAGGLE_HANDOFF.md` §2.2: **11 068 fake videos**).

| Set | Videos | Paper's size | Observed on sampled clips: size · fps · frames |
|---|---|---|---|
| T2V Moonvalley | 3 000 | 1184×672 | 1184×672 · 16 fps · **95** — *same generator as training* |
| T2V VideoCraft | 1 500 | 1024×576 | 1024×576 · 8 fps · **16** |
| T2V Pika | 1 000 | variable | 1088×640 · 24 fps · 72 |
| T2V NeverEnds | 1 000 | 1024×576 | 1024×576 · 10 fps · 30 |
| T2V Emu | 900 | 512×512 | 512×512 · 16 fps · 65 |
| T2V VideoPoet | 120 | 512×896 | 512×896 · 8 fps · 17–41 |
| T2V Hotshot | 500 | 672×384 | 672×384 · 8.33 fps · **8** |
| T2V Sora | 48 | variable | 1280×720 · 30 fps · 600 |
| I2V Moonvalley | 1 000 | variable | 626–1368 × 626–1068 (short side ≥ 626) · 16 fps · 24–46 |
| I2V NeverEnds | 1 000 | variable | 512–1024 × 576–960 (short side ≥ 512) · 10 fps · 30 |
| I2V Pika | 1 000 | variable | 576–1152 × 640–1024 (short side ≥ 576) · 24 fps · 72 |

Paths: `gdrive:SPL3/Video-NSGVD/videos_subset/T2V/<name>_mp4/` and `…/I2V/<name>_mp4/` (Moonvalley names look like
`AAiredU_output.mp4`, Hotshot files are `gif_<uuid>.mp4` — the GIFs are already MP4 — and Sora's are named after
the prompt). The I2V sets are also on this PC under `Video/Test_Videos/I2V/…`, handy for local smoke tests.

**Observed (probed 3 Oct 2026 with the local OpenCV).** The T2V rows come from the first one to three clips of each
set by name, fetched from your Drive; the I2V rows from 12 evenly spaced local clips per set. Every clip is H.264,
decodes, and its container frame count equals the decoded count. A small sample, but it already overturns an
assumption: **only Moonvalley T2V has the paper's 95 frames.** Hotshot has 8 frames (7 pairs), VideoCraft 16,
VideoPoet 17–41, NeverEnds 30, Emu 65, Pika 72, I2V Moonvalley 24–46, and Sora 600. The pair count is therefore
`min(K, pairs available)` (D8), and frame rates span 8–30 fps while the training fakes (Moonvalley) are 16 fps —
the frame-rate risk in §12 and the stride ablation of §8.6. K4 records fps, frames and size for every sampled clip.

**The authors' folder contains no real videos** (*verified*: `T2V/` = 8 zips, `I2V/` = 3 zips). The paper's reals
were GOT-10k and are not shipped.

### 3.3 Real-video pools (your decision: existing pools only)

| Pool | Source | Videos | How it is read | Role |
|---|---|---|---|---|
| R1 | HF `val/…/0_real` (YouTube-VOS, 1184×636) | 50 | the provided JPEG frames + the stored flows (Plan A) or the regenerated val flows (Plan B) — no video decoding | in-domain reals; threshold calibration; sanity |
| R2 | `real/kinetics400/videos_val/*.mp4` on Drive | 3 000 → sample | decode + RAFT | out-of-domain reals (low resolution). Probed (6 clips): 568×320, 426×320, 320×426 · 30 fps · ~300 frames · H.264 |
| R3 | `real/msrvtt/TrainValVideo/*.mp4` on Drive | 2 000 → sample | decode + RAFT | out-of-domain reals (low resolution). Probed (6 clips): 320×240 · 25–30 fps · 300–475 frames · H.264 |

Consequences to accept up front:

1. **In-domain real evidence is only 50 videos** (R1), and those same 50 also drive early stopping — their
   numbers are optimistic and their FPR estimates are noisy.
2. R2/R3 are **all below the 448-px crop on the short side** (Kinetics 320, MSR-VTT 240 — six clips of each
   probed; the earlier NSG audit also found reals ≤ 568×320), while nearly every fake is ≥ 512 px (only Hotshot,
   384, is smaller). So resolution alone separates most test reals from most test fakes: a pooled AUC would be
   inflated by that shortcut or deflated by the small-frame policy (D12), and it says nothing about 720p+ phone
   footage. Results are therefore reported **by pool and resolution bucket** (< 448 px, 448–719 px, ≥ 720 px of the
   short side — only R1 and the fakes populate the upper two), never as one pooled number, and §8.6 adds a
   resolution-matched control (fakes downscaled to the reals' size).
3. There is **no real-world (phone/YouTube) false-positive measurement**. The app must say so, and must warn when a clip is
   smaller than the training resolution (§10).
4. Comparison with paper Table III is indicative only (their reals were GOT-10k).

### 3.4 Measured while planning (read from the real HF zips and run on this PC)

**JPEG tables.** In a sample of 80 val RGB frames (40 real, 40 fake) all share a single quantisation table, which a
raw-DQT parse identifies as exactly libjpeg **quality 75** with 4:2:0 chroma; the 40 sampled flow maps (20 + 20)
are all **quality 95** (the `cv2.imwrite` default). So there is **no JPEG-quality shortcut** between the classes,
and the training inputs are Q75 RGB and Q95 flow maps. Decoded video frames are *not* JPEG, hence the
conditioning experiment (D11) with Q75 for RGB and Q95 for flow as the candidate settings (K1 repeats the audit on
a larger sample).

**CPU cost on this PC** (12 threads, fp32, torch 2.13 CPU; one measurement each):

| Operation | Time |
|---|---|
| RAFT-large 1184×640, 20 updates (the paper's setting) | **17.3 s per pair** |
| RAFT-large 1184×640, 12 updates | 11.8 s |
| RAFT-large 888×480, 20 / 12 updates | 9.7 s / 6.5 s |
| RAFT-large 592×320, 20 / 12 updates | 4.6 s / 3.1 s |
| ResNet-50 on a 448×448 crop | 0.38 s per image (batch 1 or 8) |

→ Faithful native-resolution RAFT is far too slow for the app (16 pairs ≈ 4.6 min). **Optical flow, not the CNNs,
is the CPU bottleneck**, so the flow resolution / updates / pair count must be chosen from a measured
accuracy-versus-seconds trade-off (§8.6, §9), and the flow branch must be trained on flow maps made the same
way it will be used at inference (see the flow-parity finding below).

**Flow-parity finding (⚠ the biggest technical risk).** The stored flow maps **cannot be regenerated from the
stored RGB frames.** Tried on one real and one fake val clip (flow map #0):

- RAFT-large `C_T_V1`, `C_T_V2`, `SKHT_V1` and RAFT-small, 12 and 20 updates, the official `flow_viz`: mean absolute
  difference 43 (real) and 86 (fake) on a 0–255 scale, saturation-map correlation only 0.48–0.65, and a
  decode-the-colour-wheel-and-fit analysis (all channel orders, signs and u/v transpositions, free scale and
  offset) gives R² ≤ 0.25. Frame pairs (0,1), (0,2), (0,3), (0,4), (0,6), (1,2), (2,3), … — none stands out
  (on the real clip our pair (1,2) correlates **−0.16** with the stored map #1).
- It is **not** RAFT being fragile: our flows are physically sensible (on the real clip — a car tracked by a
  panning camera — the wheels show a full rainbow rotating at about the ground speed) and very stable
  (re-encoding the frames at JPEG Q90 / Q60 or adding noise changes the magnitude map by correlation ≥ 0.98; a
  0.5×-scale flow upsampled to native size correlates 0.93–0.95 with the native one).
- The stored maps differ in *content* (wheels nearly white, a different background field) while the subject
  silhouettes coincide, so they describe the same frames. The authors' recipe for the stored maps is not
  "RAFT on adjacent frames of these JPEGs" in some way we cannot see from here.

Consequence: **we cannot assume that a flow branch trained on the stored maps accepts the maps our pipeline
makes from a video.** K2 tests exactly that, before the long training. If it fails, the flow branch is
trained on maps we regenerate ourselves (Plan B, K3b) — then training and inference share one recipe by
construction. The 0.5×-scale result also says reduced-resolution flow is structurally safe, which is what the
CPU modes in §9 rely on (still confirmed by the Kaggle ablation).

---

## 4. Design decisions and deviations from the paper

| # | Decision | Why |
|---|---|---|
| D1 | One package `AIGVDET/aigvdet/` imported by **both** Kaggle and the app; synced to Kaggle as a zip on Drive | one implementation, same constants — the lesson of `Video/PLAN.md` §7.2 |
| D2 | Our flow recipe = torchvision `raft_large` + `Raft_Large_Weights.C_T_V1` (the ported original `raft-things.pth`), 20 updates, `InputPadder`-style ×8 replicate padding, the official `flow_viz.py` (vendored, MIT, T. Runia) — i.e. the recipe of the authors' own `demo.py` | no RAFT repo to vendor, no manual Drive download; all RAFT variants tried agree with each other (§3.4), so the weights are not the issue; whether this recipe matches the *stored* training maps is decided by **G1** (D15) |
| D3 | Frames are independent training samples, exactly as the official `ImageFolder` loader; video ids are used only to group frames for video-level metrics | fidelity |
| D4 | AMP fp16 + `channels_last` + `GradScaler`, batch 64 (fallback 32 × 2 accumulation) | fp32 would triple epoch time on a T4 (*estimate* ~70 min vs ~23 min); everything else follows the official config |
| D5 | Own best-checkpoint tracker (by val ACC) plus `last.pt` | the official `EarlyStopping` is re-created after each LR drop and its first call saves "best" unconditionally, so the epoch after a drop overwrites a better checkpoint |
| D6 | The two branches train **concurrently, one per GPU**, as independent processes; resumable from Drive (`last.pt` every epoch, `--max-hours` guard) | they are independent in the paper; halves wall-clock; survives the 12 h cut-off |
| D7 | Test-time probabilities are computed **once per frame and cached** (`probs.csv`); all tables, α sweeps, K sweeps and thresholds are recomputed offline | RAFT dominates cost; never recompute it for a table change |
| D8 | Test extraction uses up to **K evenly spaced adjacent-frame pairs** inside the first 95 frames (default K = 32; sweep K ∈ {8, 16, 32, 94} on a small subset) instead of all 94; a clip with fewer pairs uses all of them (**K = min(K, pairs)**: Hotshot 7, VideoCraft 15, NeverEnds 29, I2V Moonvalley 23–45, …) | all 11 068 videos at K = 94 would be ≈ 85 GPU-h at most (*estimate*, 0.3 s/pair; less in practice because many sets are shorter) — still impossible on the free tier |
| D9 | Test set = seeded sample, ≤ 150 videos per generator set (`--quick`: 50), all of VideoPoet (120) and Sora (48) | ≈ 1 500 fake + ≈ 650 real videos ≈ 3–4 h on 2× T4 |
| D10 | Threshold: report **0.5** (code), **0.1** (paper text) and our **calibrated τ** (§8.4); the app uses the calibrated one | the paper and the repo disagree |
| D11 | Input conditioning is an **experiment**: training images are JPEG, decoded video frames are not. K5 compares raw vs JPEG-round-tripped RGB and raw vs Q95 flow maps; the winner becomes the app's preprocessing | the single most likely source of train/test mismatch |
| D12 | Resolution policy is an **experiment**: native (the authors' test protocol) vs cap at 1184×672 (never upscale) × zero-pad vs upscale for frames under 448 px | official `CenterCrop` zero-pads small frames; the cap also bounds RAFT time and memory on 1080p/4K inputs |
| D13 | **Video Forensics opens a chooser** (NSG-VD / AIGVDet); each opens the shared video-view state machine configured by a detector profile; NSG behaviour is unchanged | your decision; avoids duplicating ~400 lines of view code |
| D14 | The model is frame-wise, so in the app the K pairs are spread over the **whole clip** and shown as a timeline; benchmarks use the paper's first-95-frames window | long clips are judged on all of their length, not only the opening |
| D15 | **Flow branch, Plan A vs Plan B**, decided by the K2 compatibility test (G1): **A** = train on the stored maps (cheap, faithful) if they accept our test-time maps; **B** = train on maps regenerated by our own recipe from the HF RGB frames (K3b, ≈ 3–8 GPU-h) if they do not | the stored maps cannot be reproduced (§3.4); a flow branch must see at training time what it will see at inference |
| D16 | CPU speed settings (flow scale 0.5× / 0.75×, 12 updates, pair count) are **adopted only if the Kaggle ablation confirms they keep accuracy**; the paper's native/20-update setting stays as the `accurate` mode | flow at 0.5× correlates 0.93–0.95 with native in structure (two clips), but the classifier's tolerance is unmeasured |
| D17 | Frame rate is an **experiment**: the pair stride (adjacent = 1) is ablated at {1, 2, 3} on the 30 fps reals and the 24–30 fps fakes, emulating 30 / 15 / 10 fps. The app default stays adjacent frames (faithful); a documented heuristic stride = round(fps / 30) for clips above ~36 fps is adopted only if the ablation shows fps sensitivity, and anything above 30 fps is flagged as unvalidated | the training fakes are 16 fps (Moonvalley), the test sets run 8–30 fps, the real training clips' rate is unrecorded (probably 30), and phone clips are often 30–60 fps — motion per adjacent pair varies by up to ~7× |

---

## 5. Files to create

```
AIGVDET/
  PLAN.md                     ← this file
  README.md, STATUS.md        ← written at the end, modelled on Video/README.md and Video/STATUS.md
  aigvdet/                    ← ONE implementation, imported by Kaggle and the app
    __init__.py
    config.py                 ← constants: "inference contract" block (not tunable) + presentation block
    data.py                   ← HF index (video-id grouping), FrameFolder dataset, official augmentations, loaders
    models.py                 ← resnet50 binary head, save/load, loader for the authors' checkpoints
    flow_viz.py               ← vendored from the official repo (MIT) + flow_to_image wrapper with the channel policy
    flow.py                   ← RAFT wrapper (torchvision), InputPadder, frame pair -> flow map
    video_io.py               ← decode (cv2; GIF via PIL), pair sampling, resolution policy
    preprocess.py             ← centre-crop/pad, JPEG conditioning, normalisation (shared by val, test, app)
    fuse.py                   ← frame -> video aggregation, alpha fusion, verdict bands
    metrics.py                ← ACC/AUC/AP/TPR@FPR, bootstrap CIs, per-generator tables, threshold calibration
    train.py                  ← one-branch trainer (CLI, resumable)
  detect_aigvdet.py           ← app-facing wrapper: load_detector(), AIGVDetDetector.analyze(), plot_report(), CLI
  kaggle/
    KAGGLE_INSTRUCTIONS.md    ← cell order, what to expect, what to paste back to me
    k0_setup.py  k1_prepare_data.py  k2_flow_check.py  k3_train.py  k3b_regen_flows.py (Plan B only)
    k4_test_extract.py  k5_evaluate.py  k6_export.py      ← one file per notebook cell (existing convention;
                                                            format of the existing kaggle/KAGGLE_INSTRUCTIONS.md)
  tools/
    push_code.ps1             ← zip aigvdet/ + kaggle/ -> rclone to Drive (code sync, D1)
    pull_export.ps1           ← rclone Drive export/ -> AIGVDET/models/
  models/                     ← git-ignored: checkpoints, RAFT weights, thresholds (from Drive)
  tests/test_aigvdet_module.py
```

Edits outside `AIGVDET/` (all small, listed in §10): `app/paths.py`, **new** `app/aigvdet_engine.py`,
**new** `app/ui/video_menu_view.py`, `app/ui/video_view.py`, `app/ui/app_window.py`, `app/ui/home_view.py`,
`.gitignore` (`AIGVDET/models/`, `AIGVDET/work/`; `*.zip`, `*.pth`, `*.pdf` are already ignored).

Conventions kept: module-level constants, no dataclass config; bare names inside the module; the shared
package is imported as `aigvdet.*`, so it cannot collide with the three bare `config.py` files that
`isolated_import` already juggles.

---

## 6. Drive layout, transport, resume

```
gdrive:SPL3/Video-AIGVDet/
  code/        aigvdet_code.zip                       ← pushed by tools/push_code.ps1
  manifests/   hf_index.csv  hf_revision.json  test_manifest.csv  audit_hf.csv  bench.json  flow_diag.csv  flow_decision.json
  flows_regen/ (Plan B only) shard_*.tar of regenerated train/val flow maps + ledger.jsonl   ← ≈ 1.8 GB (every 3rd pair) to 5.4 GB
  ckpts/rgb/   last.pt  best.pt  history.jsonl  train.log      ← model+optimizer+scaler+RNG; every epoch
  ckpts/flow/  last.pt  best.pt  history.jsonl  train.log
  results/     probs/probs_<shard>.csv  ledger.jsonl  tables/*.csv  thresholds.json  figures/
  export/      aigvdet_rgb.pth  aigvdet_flow.pth  raft_large_C_T_V1.pth  aigvdet_meta.json
               parity_samples.npz  SHA256SUMS.txt
  baseline/    the authors' optical*.pth / original*.pth — strongly recommended (M0): the cheapest test of the flow recipe
```

- **Code sync (D1)**: `tools/push_code.ps1` zips `aigvdet/` and `kaggle/`, `rclone copyto` it to `code/`; K0 pulls
  and unzips it and puts it on `sys.path`. Re-run after every code change.
- **Secrets**: `RCLONE_CONF` (already exists); optional `HF_TOKEN` (the dataset is public, so only for rate limits).
- **Scratch**: `/kaggle/temp` if it exists, else `/tmp`; K0 asserts ≥ 35 GB free (HF zips 13.2 GB + extracted 13.2 GB
  peak). Nothing large goes to `/kaggle/working`.
- **Resume discipline**: training writes `last.pt` every epoch and `rclone copyto`s it; K3 starts by pulling
  `last.pt` if present. Test extraction is sharded (50 videos) with an append-only `ledger.jsonl`; a restart skips
  finished shards. A dead session costs ≤ 1 epoch or ≤ 1 shard.
- **Authors' released checkpoints (strongly recommended, M0)**: add
  https://drive.google.com/drive/folders/18JO_YxOEqwJYfbVvy308XjoV-N6fE4yP to your Drive as a shortcut (or
  `gdown --folder` in K0). They are trained on the stored maps, so running them on *our* test-time flows is the
  cheapest possible answer to the flow-recipe question (K2), and they are the baseline for G3 (§8.5).

---

## 7. Kaggle pipeline (heavy compute)

### K0 — setup (`k0_setup.py`, first cell of every notebook)
Restore `RCLONE_CONF`, install rclone, pull and unzip the code zip, print `nvidia-smi` / `os.cpu_count()` / free
scratch space and the version of every package listed in §1.1 (torch, torchvision, numpy, scipy, opencv, Pillow,
pandas, scikit-learn, huggingface_hub), assert `from torchvision.models.optical_flow import raft_large` works, and
`rclone lsd gdrive:SPL3/Video-AIGVDet` as a smoke test.

### K1 — data + audits + benchmark (`k1_prepare_data.py`) · ~30 min, CPU
1. `snapshot_download("xmulz/AIGVDet", repo_type="dataset", revision=<pinned>)` → scratch; unzip `train/` and
   `val/`; delete the zips.
2. Build `hf_index.csv` (path, kind rgb/flow, label, split, video_id, frame_idx, width, height). **Gate G0.**
3. Audits (cheap, CPU) → `audit_hf.csv` + a printed verdict: JPEG quality and chroma subsampling real vs fake,
   file-size and flow-saturation statistics real vs fake, mean absolute inter-frame difference real vs fake (a
   frame-rate proxy — the real clips' fps is not recorded, Moonvalley is 16 fps), Drive-Moonvalley-vs-HF-fake name
   overlap. If a trivial feature (e.g. JPEG quality) separates the classes, say so in the report, as the NSG
   project did for resolution.
4. Benchmark → `bench.json`: train img/s per branch at 448², batch 64, AMP; loader img/s vs worker count; RAFT
   pairs/s at 1184×640 in fp32 and fp16. These numbers fix epoch time, K and the profile choice in K3.

### K2 — flow-recipe check, **before** the long training (`k2_flow_check.py`) · ~1–1.5 GPU-h · **Gate G1**
Settles the open question of §3.4: does a flow branch trained on the stored maps accept the maps *our* pipeline
makes from a video?

1. **Re-run the planning-time diagnostics** on 20 val videos × 3 pairs (saturation-map correlation, wheel-fit
   analysis, pair offsets 0–6, RAFT variants) → `flow_diag.csv`. It must reproduce the §3.4 numbers (corr ≈ 0.5–0.65);
   if it suddenly matches, the cause is found and Plan A is simply confirmed.
2. **Compatibility test** (the real gate):
   - *preferred* (needs the authors' weights, M0): run their flow checkpoint (`optical_aug.pth`, and `optical.pth`)
     on **our** test-time flow maps (native, adjacent frames, 20 updates — the authors' `demo.py` recipe) for ≈ 100
     Moonvalley-T2V test videos (K = 16 pairs) against R1/R2 reals; also on the stored val maps as the reference;
   - *fallback* (no authors' weights): 2 epochs of `budget`-profile flow training on the stored maps, then the same
     evaluation.
3. **Decision → `flow_decision.json`**: flow-branch video-level AUC ≥ 0.85 on the Moonvalley-T2V test videos against
   the reals, and within 0.10 of its AUC on the stored val maps → **Plan A**. Otherwise → **Plan B** (K3b).
   Either way the same RAFT recipe (`aigvdet/flow.py`) is what K4 and the app use at test time.

### K3 — training, two branches in parallel (`k3_train.py` → `aigvdet/train.py`)
`subprocess.Popen` per branch (`CUDA_VISIBLE_DEVICES=0` rgb, `=1` flow), logs tailed in the cell. The **RGB branch
does not depend on the flow question and starts right after K1** (GPU 0). The flow branch starts after K2: Plan A →
immediately on the stored maps (GPU 1); Plan B → after K3b, on the regenerated maps.

- Data: `train/moonvalley_vos2_original_crop` (rgb), `train/moonvalley_vos2_crop_op` (flow; or `flows_regen/` in
  Plan B); val from `val/…`.
- Transform order as in the official `binary_dataset`: blur (p 0.1, σ 0.5) → JPEG (p 0.1, cv2 Q75), both on the
  full frame → `RandomCrop(448)` → `RandomHorizontalFlip(0.5)` → uint8 tensor; the ImageNet normalisation runs on the
  GPU (saves CPU). Loader workers = `cpu_count // 2` per process.
- ResNet-50 (ImageNet) with `fc = Linear(2048, 1)` N(0, 0.02); `BCEWithLogitsLoss`; `Adam(1e-4)`; batch 64 (fallback
  32 × 2); AMP + `channels_last`; seed 3407.
- Each epoch: train over all frames → validate on **all** val frames (centre crop, ACC@0.5, AUC, AP, plus video-level
  ACC/AUC from per-video means) → append `history.jsonl` → save/upload `last.pt`, and `best.pt` on a new best ACC.
- LR rule: ÷10 after 5 epochs without ≥ 0.001 val-ACC improvement (≥ 0.002 after the first drop); stop when the
  LR would fall below 1e-6; hard cap 40 epochs.
- Profiles: `paper` (full passes) and `budget` (32 random frames per video per epoch, ≈ 1/3 cost). K1's benchmark
  decides: if a full epoch is ≤ 30 min use `paper`; otherwise `budget`.
- *Estimate*: ~65–75 img/s per T4 at 448² under AMP → ~23 min per full epoch per branch; the paper's stopping rule
  typically needs ~20–30 epochs → 8–12 h, i.e. one to two sessions with resume (≈ 10–25 of the 30 weekly GPU-hours
  for both branches together). **Gate G2.**

### K3b — Plan B only: regenerate the flow maps ourselves (`k3b_regen_flows.py`)
Our recipe (D2) on the HF RGB frames of train and val: frames → RAFT 20 updates (adjacent pairs (j, j+1)) →
`flow_viz` → JPEG Q95, exactly like the stored maps' container → shards of 500 maps with a ledger, uploaded to
`flows_regen/` so they are never recomputed. Full set: 93 pairs × 1 100 videos = 102 300 pairs; `budget` option:
every 3rd pair (34 100 pairs, adjacent frames are highly redundant). *Estimate* 0.3 s/pair on a T4 (K1 measures) →
full ≈ 8.5 GPU-h (≈ 4.3 h on two T4s), `budget` ≈ 2.8 GPU-h. It runs on GPU 1 while the RGB branch trains on GPU 0;
the flow branch then trains on these maps, validated on regenerated val maps, and at test time the identical recipe
is used, so train and inference agree by construction. RAFT's output barely changes under JPEG re-encoding of its
input frames (§3.4), so no JPEG conditioning is applied to RAFT's inputs; the Q95 JPEG container applies only to
the finished flow map (D11).

### K4 — test extraction (`k4_test_extract.py`) · the expensive stage
- `test_manifest.csv`: seeded sample (seed 42) per set (D9), reals split by video id into `calib` / `report`
  halves (R1 is read from its provided frames/flows instead).
- Per video: decode the first 95 frames, apply the resolution policy (D12), pick `min(K, pairs)` evenly spaced
  pairs (D8); per pair: RAFT (our recipe, D2) → colour map → centre crop 448 (zero-pad if smaller);
  RGB crop from the first frame of the pair; run both ResNets on **every conditioning variant** (raw / JPEG
  round-trip; D11); store one row per (video, pair): `video_id, set, label, resolution_bucket, fps, n_frames,
  width, height, pair_idx, p_rgb_*, p_flow_*` (fps / frames / size feed the breakdowns of §8). Dual-GPU via `subprocess.Popen` with a `WORKER_SOURCE` string (the existing
  `kaggle/score_dual_gpu.py` pattern), 50-video shards + ledger, upload each shard then delete it.
- In the same pass: the authors' released weights (§8.5), if they are on your Drive.
- *Estimate*: 0.3 s/pair on a T4 (K1 measures) → K = 32 ≈ 10 s/video → ~2 200 videos ≈ 6 GPU-h on one T4, ~3 h on
  two — an upper bound, since Hotshot, VideoCraft, NeverEnds and most I2V clips have fewer than 32 pairs.
  `--quick` (50 per set, K = 16): < 1 h. Small side experiments (K sweep, resolution policy, stride, speed) run on
  ≤ 30 videos per set.

### K5 — evaluate (`k5_evaluate.py`) · CPU only, minutes
Everything in §8, computed from the cached probabilities. The flow branch's own test AUC (flow-only rows) is the
final confirmation of the K2 decision: a flow-only AUC far below the RGB-only AUC on Moonvalley T2V means the
flow recipe is still wrong and Plan B (or its variants) is needed before any app work.

### K6 — export (`k6_export.py`)
Slim state-dicts (`aigvdet_rgb.pth`, `aigvdet_flow.pth`, fp32, ~94 MB each), `raft_large_C_T_V1.pth` (~21 MB),
`aigvdet_meta.json` (thresholds, policies, val/test metrics, training config, HF revision, code hash),
`parity_samples.npz` (5 videos' frames and Kaggle's per-frame probabilities), `SHA256SUMS.txt` →
`export/`; reload every file once to verify.

---

## 8. Evaluation protocol

1. **Validation (in-domain)** — HF val, 50 + 50 videos: frame- and video-level ACC/AUC per branch and fused.
   Reported separately and labelled optimistic (val drove early stopping).
2. **Test tables** (paper-style): one row per generator set; for each: n fake, n real in the pool, **ACC@0.5,
   ACC@0.1, balanced ACC@τ, AUC, AP, TPR@5 %FPR**, each for RGB-only / flow-only / fused, with 95 % bootstrap CIs
   resampled over videos. Sora (n = 48) and VideoPoet (n = 120) carry their n.
3. **Reals by resolution bucket** (< 448, 448–719, ≥ 720 px short side) and by pool (R1/R2/R3) — FPR reported
   separately.
4. **Threshold policy**: τ_0.5, τ_0.1 and τ_cal = threshold giving 5 % FPR on the `calib` reals (R1 + calib halves
   of R2/R3 at the winning resolution policy), verified on the `report` halves. Rule: ship τ_cal if the mean
   fake recall at τ_cal is ≥ 60 %; otherwise ship τ from R1 alone and make the low-resolution warning prominent.
   Verdict bands (AI-generated / Uncertain within ±0.10 / Consistent with real) as in `Video/config.py`.
5. **Baseline (strongly recommended, M0)**: the authors' `optical_aug.pth` + `original_aug.pth` through the same
   harness (**Gate G3** — proves the harness reproduces paper-level numbers independently of our own training;
   without it, G3 is skipped and said so).
6. **Ablations** (offline from the cache unless noted): α ∈ {0, 0.25, 0.5, 0.75, 1}; K ∈ {8, 16, 32, 94};
   raw vs JPEG-conditioned inputs; resolution policy (re-runs RAFT, ≤ 30 videos per set). **Speed ablation (D16)**:
   flow scale {1, 0.75, 0.5} × updates {20, 12} — accuracy of the *same* trained branches on ≤ 30 videos per set,
   paired with the per-pair CPU seconds of §3.4 → the Pareto table that fixes the `quick` / `standard` /
   `accurate` modes in §9. **Frame-rate ablation (D17)**: pair stride {1, 2, 3}, emulating 30 / 15 / 10 fps, on the
   30 fps reals and the 24–30 fps fakes. **Resolution-matched control**: downscale a sample of fakes to the reals'
   size (320 / 240 px short side) and pass them through the identical small-frame policy — it shows how much of the
   pooled real-vs-fake separation of §8.3 is resolution alone.
7. What we do **not** claim: real-world accuracy on phone/YouTube footage (no such data), generalisation beyond
   the 2023–24 generators in GVD, or equality with the paper's numbers (different reals).

---

## 9. Local inference (CPU, `aigvdet/` + `detect_aigvdet.py`)

```
clip ─► decode ─► pick K adjacent pairs (spread over the whole clip) ─► resolution policy
     ─► RAFT (torch, CPU) ─► colour map (recipe D2) ─► crop 448 ─► ResNet-50 flow ─┐
     └────────────────────────► crop 448 (+ JPEG conditioning if it won) ─► ResNet-50 RGB ─┤─► fuse ─► verdict
```

- API mirrors `Video/detect_nsgvd.py`: `load_detector()`, `AIGVDetDetector.analyze(path, mode)`,
  `plot_report(result, out_png)`, CLI (`--plot`, `--json`, `--mode`, `--parity`).
- Files from Drive via `tools/pull_export.ps1` into `AIGVDET/models/` (~210 MB); the loader fails with a clear
  "run pull_export" message and checks sizes/SHA-256 before `torch.load` (the lesson of the truncated NSG UNet).
- **Modes** — a mode is a triple (flow setting, RAFT updates, K pairs). Measured costs on this PC (§3.4; RAFT
  per pair varied 8–17 s at 1184×640 between runs with the laptop's thermal/power state, so G6 re-measures):
  ResNet-50 0.38 s per crop (2 crops per pair), RAFT 17.3 / 9.7 / 4.6 s per pair at native / 0.75× / 0.5× scale
  with 20 updates (11.8 / 6.5 / 3.1 s with 12). Provisional table — the Kaggle speed ablation (§8.6) decides which
  settings keep their accuracy, and G6 fixes the final numbers:

  | Mode | Flow setting | K | ≈ RAFT | ≈ ResNets | ≈ total |
  |---|---|---|---|---|---|
  | `quick` | 0.5× scale, 12 updates | 8 | 25 s | 6 s | **~30 s** |
  | `standard` | 0.75× scale, 12 updates | 12 | 78 s | 9 s | **~1.5 min** |
  | `accurate` | native, 20 updates (the paper's setting) | 8 | 138 s | 6 s | **~2.5 min** |

  A reduced-scale flow field is computed, upsampled (×1/scale on the vectors) to native size and coloured there,
  so the crop geometry is unchanged. Whether a flow branch trained on native-scale maps tolerates it is exactly
  what the ablation measures.
- Memory: the RAFT correlation volume grows with (H·W)²; the resolution cap (D12) keeps it near 0.75 GB at
  1184×640 and prevents a 1080p/4K clip from needing > 5 GB.
- Result dict (UI-ready, same keys as NSG where they overlap): `p_fake`, `p_spatial`, `p_temporal`, `alpha`,
  `threshold`, `threshold_source`, `verdict`, `verdict_level`, `is_flagged`, `pairs_used`, `pair_indices`,
  `frame_probs{spatial,temporal}`, `fps`, `resolution`, `codec`, `duration_seconds`, `mode`, `seconds`, `caveats`,
  sample crops for the report figure. Errors: clips with fewer than 5 decodable frames (< 4 pairs) are rejected
  with a clear message, and fewer than 16 pairs shows a "short clip" warning — Hotshot-style 8-frame clips are
  valid inputs, so the NSG module's 8-frame minimum is not copied.
- `plot_report`: (a) per-pair spatial and temporal probabilities along the clip (timeline), (b) the RGB and flow
  crops the networks saw for the most / least suspicious pairs, (c) fused probability against τ.

---

## 10. App integration

1. `app/paths.py`: add `AIGVDET_DIR`.
2. **New** `app/ui/video_menu_view.py` — the chooser opened by the existing Home button: two cards,
   "NSG-based (diffusion physics)" and "AIGVDet-based (RGB + optical flow)", each showing model readiness and the
   typical runtime; Back to Home.
3. `app/ui/video_view.py`: add a *detector profile* (engine attribute, header title, instruction text, limitation
   text, measurement rows, figure caption, progress style). The default profile reproduces today's NSG behaviour
   exactly; the AIGVDet profile adds the per-branch probabilities, the pairs-analysed line, a **determinate
   progress bar**, and a banner when the clip's short side is below the training resolution.
4. **New** `app/aigvdet_engine.py` — copy of `video_engine.py`'s shape (`load_async`, `analyze_async` with
   identity `request_id`, one video at a time, `new_run_dir`) plus an `analysis_progress` event `(done, total)`.
   `app/ui/app_window.py` constructs it, calls `load_async()`, adds it to the `_poll_events` tuple and gains
   `show_video_menu()` / `show_aigvdet()`.
5. `app/ui/home_view.py`: the Video Forensics button now calls `show_video_menu`; `_status_text` adds
   "AIGVDet models ready".
6. NSG-VD files under `Video/` are not modified.

---

## 11. Verification gates

| Gate | Where | Pass criterion |
|---|---|---|
| G0 data | K1 | exactly 500 + 500 train and 50 + 50 val videos; 95 RGB and 93 flow images per video; every image ≥ 448 px both sides; no ID overlap between HF fake names and the Drive Moonvalley names |
| G1 flow-recipe compatibility | K2 | a flow branch trained on the stored maps reaches video-level AUC ≥ 0.85 on **our** test-time maps for Moonvalley-T2V vs reals, within 0.10 of its AUC on the stored val maps → Plan A; otherwise Plan B (K3b). The decision is written to `flow_decision.json` |
| G2 training sanity | K3 | a 100-step overfit run converges; no NaN under AMP; after epoch 1 val ACC > 0.7 on both branches |
| G3 harness validity | K5 | authors' weights (if added, M0) reproduce paper-level AUC (T2V avg ≥ 0.90) through our harness; skipped, and said so, if they are absent |
| G4 results | K5 | Moonvalley T2V (same generator) video-level AUC ≥ 0.95; T2V average AUC ≥ 0.85; flow-only AUC not far below RGB-only (else the flow recipe is still wrong). If lower: debug the recipe and policies (D2/D11/D12) before any app work |
| G5 Kaggle ↔ local | local | 5 clips, same pairs: \|Δ p_fused\| ≤ 0.03 and identical verdicts (fp32 CPU vs GPU) |
| G6 CPU timing | local | per-mode seconds recorded on this PC; `standard` target ≤ 90 s (revise the K counts, not the model, if missed) |
| G7 app smoke | local | Video Forensics → chooser → AIGVDet → pick a clip → determinate progress → verdict, figure, run dir under `results/app/`; NSG path still works end to end |
| G8 tests | local | `AIGVDET/tests/test_aigvdet_module.py` and the existing `Video/tests/test_video_module.py` both pass |

Unit tests (no weights needed): `flow_viz` against the vendored official output; `InputPadder`; centre-crop/zero-pad;
pair sampling; α-fusion and verdict bands; HF filename parser (`rsplit`); official augmentation order; a 2-step
synthetic training run on CPU; authors'-checkpoint key compatibility.

---

## 12. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **The stored flow maps cannot be reproduced** (§3.4) — a flow branch trained on them may not accept our test-time maps | silent flow-branch failure, 8+ GPU-h wasted | K2 / G1 *before* the long training, using the authors' weights as a cheap oracle (M0); Plan B regenerates training maps with our own recipe (K3b); flow-only AUC re-checked in K5/G4. Last resort if no flow recipe works: ship the RGB branch alone (α = 1; paper Table II: 86.5 % ACC / 94.9 AUC) and say so in the UI |
| RAFT too slow on this CPU (measured 8–17 s per native-resolution pair) | app feels broken | three modes (§9); flow scale, updates and pair count are the levers, chosen by the Kaggle speed ablation; G6 re-measures; ONNX/OpenVINO only as a follow-up |
| Train/test input mismatch (JPEG frames vs decoded video; low-res clips) | high false positives on real videos | D11/D12 experiments; resolution warning in the app; reals reported by resolution bucket |
| Only 50 in-domain reals, no real-world reals | threshold and FPR estimates are weak | R1 + R2/R3 reported separately; conservative threshold rule (§8.4); explicit caveat text in the UI |
| Every R2/R3 real is < 448 px (Kinetics 320, MSR-VTT 240) while nearly every fake is ≥ 512 px | resolution separates the classes in the test pool; pooled AUC is misleading | by-pool and by-bucket reporting with no pooled headline; resolution-matched control (§8.6) |
| Frame rate differs by class and generator (Moonvalley fakes 16 fps; the real training clips' rate is unrecorded, probably 30; test sets 8–30 fps; phone clips often 30–60 fps) | the flow branch may key on motion scale rather than generation artefacts | K1 inter-frame-difference audit; stride ablation (D17); fps shown in the report; caveat in the UI |
| Single-generator, single-real-source training (Moonvalley vs YouTube-VOS) | content/style shortcuts (e.g. static real scenes give near-zero flow, like I2V fakes) | per-generator rows; honest caveats; optional follow-up: more diverse training data (needs your approval) |
| Kaggle 12 h cut-off / session loss | lost GPU hours | per-epoch `last.pt` on Drive, `--max-hours`, sharded extraction with ledger |
| Disk: 26 GB peak on scratch | job dies late | K0 asserts free space; delete zips after unzip; nothing large in `/kaggle/working` |
| Paper vs code vs README disagree (threshold 0.1/0.5; 94/93 flows; PNG/JPEG; folder layout) | wrong assumptions | the verified data facts in §3 are the source of truth; thresholds reported both ways |
| HF repo changes or disappears | not reproducible | pinned revision `c1326bef…`; recorded in `hf_revision.json` |
| Dataset licence (academic research only) | misuse | recorded in `README.md`; cite both papers |
| AMP differences from the paper's fp32 | small accuracy delta | validated by G2 and by G3 (authors' fp32 weights through our harness) |

---

## 13. Milestones — who does what

| # | Milestone | Where | Who |
|---|---|---|---|
| M0 | Create the Drive folder tree; **add the authors' checkpoint folder to your Drive** (strongly recommended — it is the cheap oracle for the flow recipe and the G3 baseline) | Drive | you |
| M1 | `aigvdet/` package + unit tests + `tools/push_code.ps1` (gate G8 partial) | local | Claude |
| M2 | K0–K1 (bootstrap, data audit, benchmark; G0), then **K2 flow-recipe check (G1)** → Plan A or B; you paste back the printed summaries | Kaggle | you run, Claude reviews |
| M3 | K3 (+ K3b under Plan B): both branches trained (G2); the RGB branch can start as soon as K1 is done; you paste back `history.jsonl` tails | Kaggle | you run, Claude monitors |
| M4 | K4–K6: test extraction, tables, speed ablation, thresholds, export (G3, G4) | Kaggle | you run, Claude analyses |
| M5 | `tools/pull_export.ps1`, local `detect_aigvdet.py` + CLI; G5, G6 | local | you download, Claude implements |
| M6 | App: chooser, engine, profile, progress bar (G7) | local | Claude |
| M7 | `README.md`, `STATUS.md`, caveat text from the real numbers (G8) | local | Claude |
| M8 | Optional: fast-flow mode (crop-first RAFT with approximated normalisation), diverse-data retrain, ONNX | — | only with your approval |

---

## Appendix A — Fixed hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Backbone | ResNet-50 ImageNet (`IMAGENET1K_V1`), `fc: 2048→1`, N(0, 0.02) | paper §III-A, `get_network` |
| Crop | train `RandomCrop(448)`, eval `CenterCrop(448)` (zero-pad if smaller) | `binary_dataset` |
| Normalisation | ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | `binary_dataset` |
| Augmentation | blur p 0.1 σ 0.5 · JPEG p 0.1 cv2 Q75 · flip p 0.5 | `config.py`, `datasets.py` |
| Loss / optimiser | `BCEWithLogitsLoss` · Adam(1e-4, β 0.9/0.999), no weight decay | `trainer.py` |
| Batch | 64 (AMP; fallback 32 × 2) | `config.py` |
| LR schedule | ÷10 after 5 epochs w/o ≥ 0.001 val-ACC gain (0.002 after first drop); stop below 1e-6; cap 40 epochs | `train.py`, `earlystop.py`, paper §III-A |
| Seed | 3407 | `config.py` |
| RAFT | `raft_large`, `C_T_V1`, 20 updates, pad ×8 replicate, flow map normalised by max radius | `demo.py`, `flow_viz.py` |
| Fusion | α = 0.5, video score = mean over frames | paper Eq. 4–5, `test.py` |
| Frames per clip | first 95 frames (benchmarks); K adjacent pairs (D8) | paper §III-A |
| Test sampling | seed 42, ≤ 150 videos per set, K = 32 | D8, D9 |

## Appendix B — Commands

```powershell
# local: push code to Drive (after every change), pull the trained export
.\AIGVDET\tools\push_code.ps1
.\AIGVDET\tools\pull_export.ps1

# local: tests, one clip, the app
venv312\Scripts\python AIGVDET\tests\test_aigvdet_module.py
venv312\Scripts\python AIGVDET\detect_aigvdet.py my_clip.mp4 --mode standard --plot
venv312\Scripts\python -m app
```
```python
# Kaggle cell 0 (every notebook): paste kaggle/k0_setup.py; then run k1 ... k6 in order
```

## Appendix C — Dependencies

Local: nothing new — everything the plan needs is already in `venv312` (exact versions and the drift notes in
§1.1). Kaggle: everything needed is preinstalled (torch, torchvision, opencv, scipy, scikit-learn, pandas); K0
prints the live versions against §1.1 and adds `huggingface_hub` only if the preinstalled one is too old.

## Appendix D — Citations and licences

- Bai, Lin, Cao (Lou, PRCV version), *AI-Generated Video Detection via Spatial-Temporal Anomaly Learning*,
  PRCV 2024 — the method and the GVD dataset. **Code and dataset are released for academic research only;
  commercial use is prohibited** — state this in `AIGVDET/README.md`.
- Teed & Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020 (torchvision port, BSD-3).
- `flow_viz.py` — Tom Runia, MIT licence (vendored with attribution).
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
