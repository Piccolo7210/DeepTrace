# Video Forensics Module — NSG-VD Implementation Plan

Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection (NeurIPS 2025)
Paper: `Video/Physics-Driven Ai video detection.pdf` · Reference code: https://github.com/ZSHsh98/NSG-VD

---

## 1. Context

The project already ships two forensic modules:

| Module | Location | Heavy compute | Local inference |
|---|---|---|---|
| Image (AEROBLADE) | `src/` | `kaggle/*.py` (2×T4, CSV out) | `src/detect.py`, CPU |
| Speech (SyncNet + AR) | `Speech/` | `Speech/kaggle/speechcode.ipynb` | `Speech/detect_pretrained.py`, CPU |
| **Video (NSG-VD)** | **`Video/` — this plan** | **`Video/kaggle/`** | **`Video/detect.py`, CPU** |

`app/ui/home_view.py` already has a "Video Forensics (coming soon)" button wired to the placeholder view. This
plan fills that slot.

**Constraints that shape every decision below**

- **Local machine**: Intel i5-1335U, 16 GB RAM, **no NVIDIA GPU**, ~20 GB free on `E:`. All local work is CPU.
- **Kaggle**: 2×T4 (16 GB each), ~30 GPU-h/week, 12 h max session, `/kaggle/working` capped at 20 GB.
- **Google Drive**: 100 GB+ available → it is the single source of truth for every artifact produced on Kaggle.
- **Dataset**: **AIGVDet / GVD** — 10 000 videos (5 000 real + 5 000 fake) drawn from the authors' Google
  Drive video folder, with `xmulz/AIGVDet` on HuggingFace as a secondary source. Academic-use-only. See §4.

**Outcome**: a local CLI `python -m Video.detect --video clip.mp4` and an in-app Video Forensics tab that
classify a video as real/fake, with all training and feature extraction done on Kaggle and checkpointed to Drive.

**The deliverable is the demo, not the benchmark table.** AIGVDet/GVD (§4) exists in this plan purely to
train, validate and test the method. What gets shown is *an arbitrary clip the user picks* — a phone video, a
YouTube download, a Sora sample — analysed locally. Every design choice below is therefore judged twice: does
it reproduce the paper, **and** does it survive a clip that resembles nothing in the training set (§7.5).
Where the two conflict, the demo wins.

---

## 2. Method summary (what we actually implement)

For a video `x` with `T = 8` uniformly sampled frames, per frame `t`:

```
NSG:   g(x, t) = ∇ₓ log p(x, t) / ( −∂ₜ log p(x, t) + λ )

spatial:    ∇ₓ log p(x, t) ≈ s_θ(x_t)                 # score from pre-trained diffusion model
temporal:   −∂ₜ log p(x, t) ≈ s_θ(x_t) · Δx_t/Δt      # brightness-constancy (Prop. 1)

⇒      g(x, t) = s_θ(x_t) / ( ⟨s_θ(x_t), Δx_t⟩ + λ )
```

- `s_θ` = OpenAI **guided-diffusion `256x256_diffusion_uncond.pt`** (~2.06 GB, ~552 M params, frozen).
  The model is ε-prediction with `learn_sigma=True` (6 output channels) → keep the **first 3 channels**, then
  `score = −ε / sqrt(1 − ᾱ_t)`. Diffusion timestep **t = 5/1000** (repo: `t_value = (diffuse_steps+1)/1000`).
- `Δx_t` = **central difference** `(x_{t+1} − x_{t−1})/2`, forward/backward at the boundary frames.
- The denominator is **reduced over (C, H, W)** to one scalar per frame (repo `.sum(dim=(2,3,4))`), so
  `g` keeps shape `(T, 3, H, W)`. `λ = 1e-10`.
- Detection: **MMD²** between the NSG features of a fixed **reference set of n = 100 real videos** and the single
  test video, under a **deep kernel**
  `k_ω(a,b) = [(1−ε)·κ(φ(a), φ(b)) + ε] · Φ(a, b)` with Gaussian `κ, Φ`.
  `φ` = one Swin block (`SingleSwinBlockDiscriminator`) → 300-d, **~0.25 M trainable params**; the frozen
  diffusion model supplies everything else.
- Verdict: `fake if MMD² > τ`. Paper default `τ = 1.0` (stable over `[0.7, 1.1]`); we additionally calibrate τ at
  a **target FPR of 5 %**, matching the project's existing stance in `models/threshold.json`.
- Kernel training objective: **MMD-MP** (multi-population aware), `J(ω) = MPP_u / sqrt(σ̂² + λ)`, Adam,
  `lr 1e-4`, `weight_decay 0.1`, `batch 24`, `σ_φ = 0.1`, `σ_Φ = 100` (verify against repo config defaults:
  `sigma=1000, sigma0=0.1, epsilon=10` are the *pre-sigmoid/pre-exp* parameterisations).

**Resolutions.** Score model runs at `resolution_size` (256 default), feature is then resized to **224×224** for
the kernel. Paper Table 7: 128×128 cuts inference time 67.7 % and keeps 98.6 % AUROC; 64×64 cuts 91.7 % and
keeps 94.6 %. A kernel checkpoint and reference set are **resolution-specific** — we produce a 256 set
(primary, accuracy) and a 128 set (optional, for a local "fast mode").

---

## 3. Repository layout to be created

```
Video/
  PLAN.md                     ← this file
  README.md                   ← written at the end, modelled on Speech/README.md
  requirements-video.txt      ← decord/av/einops/timm extras not in root requirements.txt
  config.py                   ← module-level constants (mirrors Speech/config.py; DEVICE="cpu")
  nsg/
    __init__.py
    score_model.py            ← builds guided-diffusion UNet, loads ckpt, score() helper
    nsg_features.py           ← frame sampling → NSG tensor (the single shared implementation)
    deep_kernel.py            ← SingleSwinBlockDiscriminator + deep_MMD + mmd_utils (ported)
    mmd.py                    ← MMD² of reference-set vs one test video
  vendor/
    guided_diffusion/         ← vendored from openai/guided-diffusion (MIT) — UNet + diffusion only
    NOTICE.md                 ← licences/attribution for guided-diffusion and NSG-VD code
  detect.py                   ← local CLI + detect() returning the report dict
  report.py                   ← matplotlib figure, same look as src/detect.py / Speech
  models/                     ← git-ignored; filled from Drive (see §7)
  kaggle/
    KAGGLE_INSTRUCTIONS.md
    k0_setup_drive.py         ← rclone/Drive bootstrap + ledger helpers (pasted into every notebook)
    k1_extract_frames.py      ← videos → 8-frame uint8 shards
    k2_extract_nsg.py         ← frames → NSG fp16 shards (dual-GPU subprocess pattern)
    k3_train_kernel.py        ← MMD-MP deep kernel training
    k4_reference_and_calibrate.py
    k5_evaluate_and_export.py
  tests/
    test_nsg_math.py, test_parity.py, sample_clips/
```

Conventions kept from the existing modules: **module-level constants, no dataclass config**; paths derived from
`_MODULE_ROOT = Path(__file__).resolve().parent`; bare-name imports inside the module (the app already has
`isolated_import` in `app/paths.py` to keep `src/`, `Speech/` and `Video/` from clashing on `config.py` /
`detect.py`).

---

## 4. Data plan — AIGVDet / GVD

Dataset: **`xmulz/AIGVDet`** on HuggingFace (13.2 GB), the benchmark from *AI-Generated Video Detection via
Spatio-Temporal Anomaly Learning* (PRCV 2024, arXiv 2403.16638). **Academic research use only, no commercial
use** — record this in `Video/README.md` and cite the AIGVDet paper alongside NSG-VD.

**Scope**: this dataset is used *only* for training, validation and testing. It never appears in the demo
path. Its quirks (JPEG preprocessing, 95-frame clips, watermark crops) matter exactly to the extent that they
would make the trained kernel brittle on an unseen clip — which is why §4.3 treats them as correctness bugs
rather than trivia.

### 4.1 Sources

| Where | Content | Format |
|---|---|---|
| [Google Drive folder](https://drive.google.com/drive/folders/1D84SRWEJ8BK8KBpTMuGi3BUM80mW_dKb) | **the primary source**: all GVD generated videos except AIGVDet's own 550 training clips, plus real videos from **GOT-10k** | native **video files** (MP4; Hotshot is GIF) |
| HF `train/` + `val/` | 550 real (YouTube-VOS2) + 550 fake (Moonvalley T2V) | PNG frames, ~95/video, already watermark-cropped and JPEG-processed |

**We build all splits from the Drive video files, not the HF PNGs.** Three reasons: the 10 000-video split
needs far more than the 1 100 videos HF holds; native files avoid AIGVDet's training-only JPEG preprocessing
(see §4.3a); and one input format keeps train, test and the demo path on identical pixels. The HF PNG frames
are kept as an **optional cross-check** (and as a source of YouTube-VOS2 reals for reference-pool diversity),
not as training data.

### 4.2 The split (10 000 videos)

Your division, with the development pool broken out into the three roles the method actually needs
(validation and the reference set are carved **out of** the 8 000 development videos, not added to them):

| Split | Real | Fake | Fake composition |
|---|---|---|---|
| Development pool | 4 000 | 4 000 | Moonvalley T2V 1 200 · VideoCraft 800 · Pika T2V 700 · NeverEnds T2V 700 · Emu 500 · VideoPoet 100 |
| → **Train** | 3 000 | 3 500 | the above, minus the validation share, kept proportional per generator |
| → **Reference pool** | 500 (+ wild, §K4) | — | real only, **held out of kernel training** |
| → **Validation** | 500 | 500 | stratified by generator over the dev mix |
| **Final test** | 1 000 | 1 000 | Hotshot 500 · Moonvalley I2V 452 · Sora 48 |
| **Total** | **5 000** | **5 000** | 4 000 + 1 000 real · 4 000 + 1 000 fake |

Every count is available: Moonvalley T2V has 3 000 left after AIGVDet's own 550; VideoCraft 1 500→800;
Pika 1 000→700; NeverEnds 1 000→700; Emu 900→500; VideoPoet 120→100; Hotshot 500→500 (all); Sora 48 (all);
Moonvalley I2V 1 000→452.

**Reference ≠ training.** NSG-VD's reference set is real-only and is what every test video is compared
against. Carving it out of the 4 000 development reals (500 reference · 500 validation · 3 000 training)
keeps validation MMD honest; reusing training reals as the reference makes validation look better than the
method is. Note this is the one place the plan restructures your division — the totals are unchanged.

**Validation comes from the train pool**, so it is split **by `video_id` and stratified by generator** — never
by clip, and never letting one video's clips straddle the boundary.

**Training on 6 generators is a deliberate deviation** from the paper (which trains on one). It suits the
method: we use **MMD-MP**, the multi-population variant built for heterogeneous fake populations, and a
demo detector should have seen more than one generator. Keep a single-generator run (Moonvalley T2V only)
as an ablation row if GPU time allows — it is the direct comparison to the paper.

**Three tiers of "unseen" in the test set** — report them as separate rows, never as one average:

| Tier | Set | What it measures |
|---|---|---|
| Easiest | **Moonvalley I2V (452)** | *seen generator, unseen modality* — training has Moonvalley T2V |
| Confounded | **Hotshot (500)** | unseen generator, but **GIF**: 256-colour palettization is a huge low-level shift a score model will catch on format alone. High recall here is partly artifact, and must be labelled as such |
| Hardest | **Sora (48)** | genuinely unseen and near-photorealistic — but n=48, so report a confidence interval |

**Real supply caveat**: 5 000 reals means essentially all of them come from GOT-10k, so the real side is
**single-domain**. Verify the actual GOT-10k count in the Drive folder at M1 before committing to 5 000. The
consequence for the demo is handled in K4: the reference pool must be deliberately widened with YouTube-VOS2
and wild clips, or every out-of-domain real video the demo sees will score as fake.

### 4.3 Two traps that must be handled

**(a) Compression / format confounds.** A diffusion score `s_θ(x)` is dominated by exactly the high-frequency
statistics that compression alters, so any encoding difference that correlates with the class label can be
learned instead of the physics — perfect validation numbers, useless demo.

Sourcing from native Drive files removes AIGVDet's training-only JPEG step (real frames JPEG, fakes randomly
re-compressed at Q70–90), which was the worst offender. What remains, and how it is handled:
1. **M1 audit**: per class and per generator, tabulate codec, bitrate, resolution and frame rate from
   `ffprobe`. If reals and fakes separate cleanly on any of these, say so in the report.
2. **Robustness augmentation in training** (K2): random JPEG Q60–95 and a resize round-trip applied
   **identically to both classes**, so compression becomes nuisance variation rather than signal.
3. **GIF** (Hotshot): a format confound that cannot be removed, only disclosed — see the tier table above.
4. **Watermarks**: Discord-sourced fakes carry them, GOT-10k reals do not. The bottom-band crop is applied to
   **every** video of **both** classes (§4.5 rule 3), otherwise the detector learns the banner.

**(b) Frame sampling / Δt consistency.** NSG's denominator is `⟨s_θ(x_t), Δx_t⟩`, so the feature **scale
depends on how far apart the sampled frames are**. Sampling "8 uniform frames across a whole 30 s video" in
one place and "8 uniform across 95 frames" in another puts them on different motion scales and quietly
destroys the MMD comparison.

**Canonical sampler (one rule — training, reference set, test, and the local app):**
```
take the first min(N, 95) frames of the clip → sample 8 uniformly spaced indices from that window
```
`Video/config.py`: `FRAME_WINDOW = 95`, `T = 8`; asserted in the parity test (§8 gate 4). Clips with fewer
than 8 frames are dropped; 8–94 frames are sampled uniformly over what exists. The demo's multi-window mode
(§7.4) applies this same rule inside each window, so every input the kernel ever sees has one shape and one
motion scale.

**One clip per video** in every split — with 10 000 videos there is no need for the multi-clip augmentation
that a 1 100-video source would have required, and one-clip-per-video keeps train/val/test independence
trivially clean.

### 4.4 Acquisition on Kaggle

**Videos (primary).** Copy the authors' Drive folder into *your own* Drive once (Drive UI → add shortcut →
copy), after which the existing `rclone` remote pulls it on Kaggle like any other artifact — no `gdown`
fragility, no repeated quota-limited downloads. Pull only the generators the split needs:
```bash
rclone copy gdrive:SPL3/Video-NSGVD/videos_subset /kaggle/tmp/gvd \
  --include "{0_real,1_fake/{Moonvalley*,VideoCraft,Pika_T2V,NeverEnds_T2V,Emu,VideoPoet,Hotshot,Sora}}/**" \
  --transfers=8 --progress
```
(exact folder names confirmed at M1 — the include filter is written from the real listing, not guessed).

**HF PNGs (optional cross-check + reference diversity).**
```python
from huggingface_hub import snapshot_download          # resumable, hash-checked
snapshot_download(repo_id="xmulz/AIGVDet", repo_type="dataset",
                  revision=PINNED_SHA,                  # pin for reproducibility
                  allow_patterns=["train/trainset_1/0_real/**"],   # YouTube-VOS2 reals only
                  local_dir="/kaggle/tmp/aigvdet", max_workers=8)
```
- Everything lands in **`/kaggle/tmp`**, never `/kaggle/working` (20 GB cap).
- Pin the HF revision SHA and the Drive folder inventory in `manifests/hf_subset_spec.json`.
- Budget: 10 000 videos ≈ 30–60 GB, inside the 100 GB Drive allowance.

### 4.5 Rules that affect correctness

1. Drop every clip with **fewer than 8 frames**.
2. **Never re-encode** except through the deliberate robustness augmentation of §4.3(a)(2) — and when it is
   on, apply it to **both** classes identically.
3. **Watermark cropping applies to every video of both classes**, train and test alike. Discord-sourced fakes
   carry watermarks and GOT-10k reals do not, so cropping only the fakes would *be* the confound. Use one
   fixed bottom-band fraction, verified visually on one clip per generator at M3 (contact sheets, K1).
4. Aspect ratios vary wildly (512×512 → 1184×672). Fix one policy — **center-crop to square, then resize to
   `resolution_size`** — and apply it identically everywhere.
5. `video_id` is stable and unique across sources: `mv_t2v_00042`, `yvos_00013`, `got10k_00311`.
6. Per-generator counts go in the manifest, not in anyone's memory.

### 4.6 Manifest

Before any GPU work, one CSV is produced and stored on Drive — it is the contract between all later stages:

`manifests/manifest_v1.csv`
```
clip_id, video_id, rel_path, kind, source, generator, modality, label, split, clip_index, n_frames, fps, width, height, sha1_16
yvos_00013_c0, yvos_00013, train/trainset_1/0_real/video_00013, frames, youtube-vos2, -,          -,   0, train, 0, 95,  -, 448, 448, 9f2a...
mv_t2v_00042_c2, mv_t2v_00042, train/trainset_1/1_fake/video_00042, frames, gvd, Moonvalley, T2V, 1, train, 2, 95,  -, 448, 448, 1c83...
got10k_00311_c0, got10k_00311, test/0_real/got10k_00311.mp4, video, got-10k, -,         -,   0, test,  0, 210, 30, 1280, 720, 77b1...
sora_00007_c0, sora_00007, test/1_fake/sora/00007.mp4,       video, gvd, Sora,       T2V, 1, test,  0, 120, 24, 1920, 1080, 4e0d...
```
`kind` = `frames` (HF PNG folder) or `video` (MP4/GIF) and decides which K1 path runs. `label`: 0 = real,
1 = fake. `split`: train / val / ref / test, **assigned per `video_id`** so clips never straddle splits.
`sha1_16` detects a silently changed input.

---

## 5. Storage & checkpointing on Google Drive

Drive is the **only** place where an artifact is considered safe. Nothing is left solely in `/kaggle/working`.

### 5.1 Drive folder tree

```
GDrive:/SPL3/Video-NSGVD/
  manifests/            manifest_v1.csv, ref_split.json, test_split.json, hf_subset_spec.json
  videos_subset/        the exact HF subset, mirrored once (reproducibility insurance)
  ledgers/              frames.jsonl, nsg_256.jsonl, nsg_128.jsonl     ← resume state, append-only
  frames/               f256/shard_0000.npz … (uint8, ~1.5 MB/video)   ← ~15 GB for 10k videos
  nsg/
    steps5_res256/      real/shard_0000.npz, fake/shard_0000.npz       ← fp16, ~2.3 MB/video ≈ 23 GB
    steps5_res128/      …                                              ← ~0.6 MB/video ≈ 6 GB
  ckpts/
    kernel_res256_seine_mp/  best_ckpt.pth, last_ckpt.pth, ckpt_ep{N}.pth, history.json
    kernel_res256_seine_mp/  training_state_ep{N}.pt   ← model+optimizer+RNG, for true resume
  reference/            ref_res256_n300.npz, ref_meta.json
  results/              val_metrics.csv, test_metrics.csv, threshold_res256.json
  export/               nsgvd_local_res256.pt (bundle), nsgvd_local_res128.pt, SHA256SUMS.txt
  logs/                 k2_20261001_0912.log, …
```

### 5.2 Kaggle ⇄ Drive transport — rclone

Chosen over PyDrive2/service accounts because service accounts have no My-Drive storage quota, and rclone gives
free resume (`copy` skips files already present with matching size/hash).

One-time setup:
1. **Locally**: install rclone, run `rclone config` → `drive` remote named `gdrive` → OAuth in the browser.
2. Copy the contents of `%APPDATA%\rclone\rclone.conf` into a **Kaggle Secret** named `RCLONE_CONF`.
3. In every notebook, cell 0 writes it back:
   ```python
   from kaggle_secrets import UserSecretsClient
   conf = UserSecretsClient().get_secret("RCLONE_CONF")
   Path("/root/.config/rclone").mkdir(parents=True, exist_ok=True)
   Path("/root/.config/rclone/rclone.conf").write_text(conf)
   ```
4. `rclone copy /kaggle/working/out gdrive:SPL3/Video-NSGVD/nsg/steps5_res256/real --progress --transfers=4`

Locally, the same remote pulls artifacts back: `rclone copy gdrive:SPL3/Video-NSGVD/export Video/models`.
(Google Drive for Desktop is a fine manual alternative; `.gitignore` must gain `client_secrets.json`,
`credentials.json`, `rclone.conf`.)

### 5.3 Resume discipline (applies to every Kaggle stage)

- Work is chunked into **shards of 500 videos**, named deterministically from the manifest order.
- Loop: *shard exists in ledger → skip* → compute → write to `/kaggle/working` → `rclone copy` → append a
  ledger line → delete local shard (keeps `/kaggle/working` under its 20 GB cap).
- Ledger line: `{"shard": "real/shard_0007", "n": 500, "bytes": 1180000000, "sha256": "...", "done_at": "..."}`.
- Training additionally saves `training_state_ep{N}.pt` with **model + optimizer + scheduler + RNG state**
  every epoch (the Speech module saves optimizer state but never reloads it — we fix that here, so a
  12 h session cut-off costs at most one epoch).
- A session that dies mid-shard loses ≤ 500 videos of work (~4 minutes).

---

## 6. Kaggle pipeline (heavy compute)

Following the existing `kaggle/` convention: **one `.py` file per notebook cell**, dual-GPU via
`subprocess.Popen` on a worker file written out from a `WORKER_SOURCE` string (not `multiprocessing` — spawn
cannot pickle notebook-defined functions), progress printed every 200 items with rate + ETA.

### K0 — environment & Drive bootstrap (`k0_setup_drive.py`)
- `pip install -q einops timm blobfile decord huggingface_hub` (torch/torchvision are preinstalled on Kaggle).
- Install rclone, restore `RCLONE_CONF`, `rclone lsd gdrive:SPL3/Video-NSGVD` as a smoke test.
- `rclone copy` the GVD videos from Drive per §4.4 into `/kaggle/tmp` (+ the HF YouTube-VOS2 reals for the
  reference pool); build/refresh `manifest_v1.csv` implementing the §4.2 split; mirror the manifest to Drive.
- Run the **ffprobe/encoding audit** of §4.3(a)(1) here (cheap, CPU) → `results/ffprobe_audit.csv`, and the
  **inventory check**: per-generator and GOT-10k real counts must support the §4.2 split before K1 starts.
- Download `256x256_diffusion_uncond.pt` once
  (`https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt`) and
  **mirror it to Drive** so later sessions pull it from there instead of re-downloading.
- Verify manifest ↔ dataset: every `rel_path` exists, `n_frames ≥ 8`.

### K1 — clip extraction (`k1_extract_clips.py`) · CPU-bound, ~2–3 h for 10 000 videos
One input path, one output format — which is what keeps train, test and the demo comparable:

- Decode the first 95 frames (decord → PyAV → cv2 fallback; **imageio for GIF/Hotshot**), honour rotation
  metadata, apply the **canonical sampler** (§4.3b) and the **watermark crop** (§4.5 rule 3).
- Then: center-crop to square → resize to 256×256 bilinear → store `uint8` `(8, 3, 256, 256)` per clip,
  500 clips per `np.savez` shard (≈ 750 MB).
- The HF PNG folders use the same sampler over `video_XXXXX/*.png`, and are only pulled for the
  YouTube-VOS2 reference reals and the optional cross-check.
- **Why cache clips**: NSG features are recomputed whenever resolution, timestep or augmentation changes;
  decoding is the slow, non-deterministic part, and caching makes K2 restartable and every A/B an exact
  comparison on identical pixels.
- Emit a **contact sheet per generator** to Drive — the cheapest way to catch a wrong watermark crop, a GIF
  decoded as one frame, a sideways clip, or a squashed aspect ratio before spending GPU hours.
- Also emit `results/ffprobe_audit.csv` (codec, bitrate, fps, resolution per class/generator) for §4.3(a)(1).

### K2 — NSG feature extraction (`k2_extract_nsg.py`) · **the expensive stage**
- Load the frozen UNet once per worker in **fp16**, `torch.no_grad()`, batch 8–16 videos (64–128 frames).
- Per batch: `x ∈ [0,1] → 2x−1` → diffuse to `t=5/1000` with a **fixed per-video RNG seed** (seeded from
  `video_id`, so the feature is reproducible and Kaggle/local parity is testable) → UNet → split first 3
  channels → `score = −ε/sqrt(1−ᾱ)` → central-difference `Δx` → NSG → resize to 224 → cast fp16.
- Output `(8, 3, 224, 224)` fp16 per clip, sharded and uploaded exactly as in §5.3.
- **Budget**: paper reports ≈ 0.36 s/clip on a 3090 at 256. On 2×T4 fp16 expect ≈ 0.6–1.0 s/clip.
  Workload: 10 000 clips (one per video) ≈ **2–3 GPU-h at res 256**; + ~2 GPU-h for the 2 robustness-augmented
  copies of the 7 500 training clips; + ~1 h for the optional res-128 pass. Total ≈ **5–6 GPU-h**, spread over
  2 sessions, inside the 30 h weekly quota.
- **Storage**: 10 000 × 2.3 MB ≈ 23 GB at res 256, plus ~17 GB of augmented training features ≈ **40 GB** on
  Drive. Within the 100 GB budget, but it is the largest single consumer — shard-and-upload discipline (§5.3)
  matters here more than anywhere else.
- **Robustness variants for the training split only** (this is what makes the demo work): extract features not
  just from the clean clip but from 2 augmented copies — random JPEG quality 60–95, and a resize round-trip
  (down to 0.6× and back) — applied **identically to both classes**. The kernel then sees compression and
  rescaling as nuisance variation rather than as the class signal. Cost: ~2× the training-split extraction
  (~+30 GPU-min), and it directly buys demo-domain generalisation. Never augment the reference pool or test.
- Sanity gate before the full run: extract 50 videos, assert no NaN/Inf, and plot the histogram of
  `−∂ₜ log p` — paper Fig. 5 says values should sit **outside `[−0.1, 0.1]`**. If they don't, the score scaling
  or `Δx` convention is wrong; stop and fix before burning GPU hours.

### K3 — deep kernel training (`k3_train_kernel.py`) · ~1–2 GPU-h
- Dataset = NSG shards, memory-mapped: **3 000 real + 3 500 fake** training clips (plus augmented copies);
  `batch 24`, Adam `lr 1e-4`, `wd 0.1`, up to 200 epochs with early stop.
- Objective **MMD-MP** (`is_yy_zero=True`) — the right choice here, since the training fakes span 6 generators
  (§4.2). Train the MMD-D variant and a single-generator (Moonvalley T2V only) run as ablation rows.
- Validation every epoch on the **500 real + 500 fake** validation split, scored against the held-out
  **reference pool** → **AUROC**; keep `best_ckpt.pth` by val AUROC.
- Sampling per batch should be **balanced across generators**, not uniform over clips — otherwise Moonvalley
  (1 200) dominates VideoPoet (100) by 12:1 and the kernel quietly specialises.
- Save `training_state_ep{N}.pt` each epoch and push to Drive (§5.3).
- Trainable params ≈ 0.25 M → each checkpoint is a few MB; keeping all epochs is cheap.

### K4 — reference set + threshold calibration (`k4_reference_and_calibrate.py`)
- The reference pool is the **500 reals held out of training** (§4.2), widened for domain diversity; the
  detector draws `n = 100` from it (paper's ablation: stable ≥ 50, degrades at n = 10).
- **Diversity is the point.** Our reals are essentially all GOT-10k, so a pool drawn only from them defines
  "real" as *GOT-10k*, and every out-of-domain real video scores as fake. Target mix:
  ~60 % GOT-10k · ~20 % YouTube-VOS2 (from the HF PNGs) · ~20 % wild clips. NSG-VD Table 9 shows even a 3:7
  domain mix restores performance, so this is cheap insurance.
- **Wild reals in the pool** (demo insurance): add ~50 ordinary real clips of the kind the demo will actually
  use — phone footage, YouTube downloads, screen-recorded clips at mixed resolutions and codecs. Drop them in
  a Drive folder, and K4 extracts their features like any other reference clip. A pool made only of
  YouTube-VOS2 + GOT-10k defines "real" as *those two datasets*, and every out-of-domain real video then
  scores as fake. Target mix ≈ 40 % wild / 60 % dataset.
- Calibrate τ on a validation set that **includes** wild reals, and report the FPR on wild reals separately.
  That single number predicts the demo's failure mode better than any benchmark row.
- Store the reference **NSG features** (not the videos) → `ref_res256_n300.npz` (~700 MB).
- Compute MMD scores for a validation set of real + fake videos, then produce
  `threshold_res256.json` in the same shape the project already uses for images:
  ```json
  {"threshold": 0.93, "method": "target_fpr_0.05", "tau_paper_default": 1.0,
   "auroc": 0.95, "achieved_fpr_on_real": 0.048, "tpr_on_generated": 0.94,
   "n_ref": 100, "resolution": 256, "steps": 5, "calibrated_at": "..."}
  ```
  Report both the calibrated τ and the paper's τ = 1.0 so the two are comparable.

### K5 — evaluation & export (`k5_evaluate_and_export.py`)
- Evaluate on the **2 000-video final test set** (1 000 real + Hotshot 500, Moonvalley I2V 452, Sora 48):
  **Recall, Accuracy, F1, AUROC** → `test_metrics.csv`, shaped like paper Tables 1–3, with **one row per
  generator** and the three-tier framing of §4.2 (seen-generator/unseen-modality · GIF-confounded · Sora n=48).
  Never publish only the average — it would hide exactly the distinctions that matter.
- Report **validation numbers separately from test numbers**. Validation shares generators with training and
  will look much better; presenting it as the headline would overstate the method.
- Also report the AIGVDet paper's own numbers on the same generators, so the comparison is
  physics-driven (ours) vs optical-flow CNN (theirs) on identical data — that contrast is the point of the
  module, and it is a stronger result than reproducing NSG-VD's GenVideo table.
- Baseline row: run the **authors' pretrained `standard-SEINE-mp.pth`** through the same evaluation. Note the
  domain gap — it was trained on Kinetics-400/SEINE, not GVD — so treat it as a pipeline sanity check
  (features and MMD plumbing are correct), not as a fair competitor.
- Report the **with- vs without-robustness-augmentation** runs side by side, including on re-compressed copies
  of the test set (§8 gate 6). A large gap is itself a finding worth writing up.
- Build the **local bundle** (mirrors `Speech/models/speech_forensics_model.pt`):
  ```python
  torch.save({
    "model_state_dict": kernel.state_dict(),      # Swin block + ε/σ/σ0 params
    "kernel_hparams": {...}, "config": {"resolution":256,"steps":5,"T":8,"lambda":1e-10},
    "ref_features": ref_fp16,                     # (100, 8, 3, 224, 224) fp16 ≈ 230 MB
    "ref_meta": {...}, "thresholds": {"target_fpr_05":0.93,"paper":1.0},
    "val_metrics": {...}, "trained_epoch": N, "created_at": "..."
  }, "nsgvd_local_res256.pt")
  ```
- `rclone copy` the bundle + `SHA256SUMS.txt` to `export/`.

---

## 7. Local side (inference only, CPU)

### 7.1 Files pulled from Drive into `Video/models/` (git-ignored)

| File | Size | Purpose |
|---|---|---|
| `256x256_diffusion_uncond.pt` | 2.06 GB | frozen score model |
| `nsgvd_local_res256.pt` | ~250 MB | kernel + reference features + thresholds |
| `nsgvd_local_res128.pt` | ~70 MB | optional fast mode |

Total ≈ 2.4 GB — fits in the ~20 GB free on `E:`. `detect.py` fails with a clear "run
`rclone copy gdrive:SPL3/Video-NSGVD/export Video/models`" message if a file is missing, and verifies SHA256.

### 7.2 Inference path

`Video/nsg/nsg_features.py` is **the same code that ran on Kaggle** — one implementation, imported by both
sides. Locally it runs fp32 on CPU with `torch.set_num_threads(os.cpu_count())`.

```
video → first min(N,95) frames → 8 uniform indices     ← canonical sampler §4.3(b)
      → watermark crop (opt) → center-crop square → 256×256
      → diffuse to t=5/1000 (seeded from file hash)
      → UNet ×8 frames  ← the whole cost
      → score → NSG → 224×224
      → Swin block φ → 300-d
      → MMD²(reference 100, test) → verdict vs τ
```

The sampler and crop policy live in `nsg_features.py` and are **the same constants** used on Kaggle. If a
user's video is 4K/60 fps, the first 95 frames are ~1.6 s of footage — acceptable, since NSG reads
frame-to-frame dynamics rather than scene semantics, but it must be stated in the report JSON
(`frames_window`, `duration_analysed_s`) so a verdict is never mistaken for a judgement on the whole video.

**Expected latency** (to be measured, not assumed): the UNet is ~1.1 TFLOP/forward at 256×256 and this CPU
sustains roughly 100–200 GFLOP/s in PyTorch, so **~1–3 minutes per video at 256** and **~20–45 s at 128**.
Mitigations, in order of preference:
1. Frames are processed in a batch of 8 in one pass (already the plan) — no per-frame Python overhead.
2. `--fast` flag → the res-128 bundle.
3. If still too slow, export the UNet to **ONNX Runtime** (CPU) as a follow-up; it typically gives 1.5–2× over
   eager PyTorch on this class of CPU. Not in the initial scope.

The UI must therefore show a **determinate progress bar** (8 frame-steps), and the engine must run analysis off
the Tk main thread — which `app/speech_engine.py`'s `analyze_async` pattern already does.

### 7.3 `Video/detect.py` — CLI and report

CLI flags mirror `Speech/detect_pretrained.py`: `--video`, `--bundle`, `--threshold`, `--threshold-key`,
`--resolution {256,128}`, `--fast`, `--windows {N|auto}`, `--device`, `--plot`, `--out-dir`.

Report JSON (written to `results/` as `<stem>_report.json`, figure as `<stem>_analysis.png`, same as the image
module):
```json
{"file": "clip.mp4", "mmd2": 1.47, "threshold": 0.93, "threshold_source": "target_fpr_0.05",
 "verdict": "fake", "confidence_band": "high",
 "per_frame_nsg_norm": [...8 values...], "temporal_derivative_stats": {"min":..,"max":..,"near_zero_frac":..},
 "windows_analysed": 3, "window_scores": [1.31, 1.62, 1.48], "window_times_s": [0.0, 12.4, 24.8],
 "n_reference": 100, "resolution": 256, "frames_used": 8, "frames_window": 95,
 "source_fps": 25, "duration_s": 30.2, "duration_analysed_s": 11.4,
 "runtime_s": 214.0, "warnings": ["rotation metadata applied"],
 "caveats": ["trained on GVD 2023-24 generators; newer generators may evade"],
 "figure": "...", "report": "..."}
```
Three-level verdict like the Speech module — `real` / `suspicious` / `generated` — using a band around τ, since
`widgets.make_verdict_banner` already renders exactly those three states.

`report.py` figure: frames filmstrip on top, per-frame NSG magnitude below, and the MMD score plotted against
the reference-set MMD distribution (that histogram comes from the bundle) — this makes the verdict legible
rather than a bare number.

### 7.4 Demo path — arbitrary clip in, verdict out

This is the deliverable. A demo clip is nothing like a training clip: unknown codec, resolution, frame rate,
duration, orientation, and it may be 30 seconds rather than 95 frames.

**Multi-window analysis.** Analysing only the first ~95 frames of a 30 s clip is both a weak story and a weak
signal. Instead: place `W` windows of 95 frames evenly across the clip, run the canonical sampler inside each
(so every window is exactly the in-distribution input the kernel was trained on), and aggregate.

| `--windows` | Behaviour | CPU cost at res 256 |
|---|---|---|
| 1 | first window only — quick look | ~1–3 min |
| 3 (**default for the demo**) | start / middle / end | ~3–9 min |
| `auto` | one window per ~5 s, capped at 6 | up to ~18 min |

- Aggregate by **mean MMD²** for the verdict (robust), and keep the **per-window scores** for the figure —
  a timeline of suspicion is far more convincing in a demo than one number, and it localises which part of a
  clip looks synthetic.
- Report `windows_analysed`, their timestamps, and per-window scores in the JSON.

**Input handling** (each of these is a demo that would otherwise fail in front of an audience):

| Case | Handling |
|---|---|
| MOV / WebM / MKV / AVI / GIF | decord → PyAV → cv2 → imageio fallback chain; GIF via imageio |
| Clip shorter than 8 frames | refuse with a clear message, not a crash |
| 8–94 frames | single window, sample uniformly over what exists |
| 4K / 60 fps | decode-and-downscale on the fly; never load full-res frames into RAM |
| Portrait / non-square | same center-crop-to-square policy as training (§4.5 rule 4) |
| Bottom-band crop | applied to demo clips **too** — one policy everywhere, so a user clip is framed exactly like the reference pool it is compared against (`--no-crop` to disable). It also drops burned-in subtitles and logos, which is a bonus |
| Rotation metadata | honour the display matrix — a sideways clip is out of distribution |
| Audio-only or corrupt file | validated up front, reported as an error verdict |
| Variable frame rate | index by frame, not timestamp, so sampling stays uniform |

**Honesty in the verdict.** The detector is trained on 11 generators from 2023–24. A 2026 model may sit
outside everything it has seen, so the report carries an explicit `caveats` list (`"trained on GVD generators;
newer generators may evade"`, `"only N seconds analysed"`) and the UI shows the three-level
`real / suspicious / generated` banner rather than a binary claim. A demo that says "suspicious, here's the
per-window evidence" is stronger than one that says "FAKE" and is wrong.

### 7.5 App integration

- `app/paths.py`: add `VIDEO_DIR`; keep using `isolated_import` (three modules now define `config.py`/`detect.py`).
- `app/video_engine.py`: copy `speech_engine.py` — `load_async()` on a daemon thread, `analyze_async(path)`
  returning a `request_id`, events pushed onto `self.events` as `("analysis_done", payload, request_id)`,
  run dirs via `new_run_dir(path)` → `results/app/<ts>_<stem>/`.
- `app/ui/video_view.py`: copy `speech_view.py`'s state machine (instructions → selected → running → results →
  error) plus its `_probe_video` thumbnail helper, and add the per-frame progress bar.
- `app/ui/app_window.py`: instantiate the engine, add it to `_poll_events`, add `show_video_forensics()`.
- `app/ui/home_view.py:38-44`: point the existing button at the real view; add a video entry to `_status_text`.

---

## 8. Verification

Run in this order — each gate catches a different class of bug.

1. **NSG math unit tests** (`tests/test_nsg_math.py`, local, no GPU): with a toy analytic Gaussian score
   function, assert the implementation reproduces Proposition 3/4 closed forms; assert the central-difference
   `Δx` matches a hand-computed value on a 3-frame ramp; assert `λ` prevents division blow-up when
   `⟨s, Δx⟩ = 0`.
2. **Score-model sanity** (Kaggle): denoise a noised image at t=5 and confirm it reconstructs — proves the
   ε/σ convention and channel split are right before 4 GPU-hours are spent on them.
3. **Temporal-derivative distribution** (Kaggle, 50 videos): histogram of `−∂ₜ log p` lies outside
   `[−0.1, 0.1]` (paper Fig. 5).
4. **Kaggle ↔ local parity** (`tests/test_parity.py`): same 5 clips, same seed, both sides →
   `max |Δ| < 1e-2` and cosine similarity `> 0.999` on the NSG tensors, and the **same verdict**. fp16-vs-fp32
   makes exact equality impossible; a verdict flip here means the local path is broken.
5. **Reference-implementation baseline** (Kaggle): authors' `standard-SEINE-mp.pth` evaluated by our K5
   script — a plumbing check (see K5), not a fair competitor on GVD.
6. **Robustness control** (Kaggle): evaluate the kernel trained with and without the compression/resize
   augmentation (§4.3a) on the test set *and* on re-compressed copies of it. A model that loses several points
   under re-compression will not survive the demo.
7. **Wild-real FPR** (the demo's real referee): score ~50 ordinary real clips (phone, YouTube, screen capture)
   that appear in no split. This number predicts how the app behaves on a stranger's video far better than any
   benchmark row. If it is bad, widen the reference pool (K4) before anything else.
8. **End-to-end local** on `Video/tests/sample_clips/` (2 real, 2 generated held out from every split):
   `python -m Video.detect --video ...` produces the report, correct verdicts, and a recorded runtime.
9. **App smoke test**: `venv312\Scripts\python -m app` → Video Forensics tab → pick a clip → progress bar
   advances → verdict banner and figure render → run dir written under `results/app/`.
10. **Demo rehearsal** (the gate that actually decides whether this ships): run the exact clips you intend to
    demo with — a few real phone/YouTube clips and a few current AI-generated ones (Sora/Veo/Kling samples
    from the web, i.e. generators *not* in GVD) — end to end in the app. Record verdict, score and runtime for
    each. Two honest outcomes to prepare for: newer generators may evade a detector trained on 2023–24
    models, and long clips take minutes. Both are manageable if known in advance; neither is manageable if
    discovered live.

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| CPU inference slower than estimated (>5 min/video) | app feels broken | measure at step 6 before app work; ship `--fast` res-128 bundle; ONNX Runtime as follow-up |
| **Encoding/format correlates with the class label** | inflated metrics, weak real-world detector | source native Drive files (not the JPEG-processed HF frames); ffprobe audit at M1; robustness augmentation in K2 (§4.3a) |
| **Hotshot is GIF** — palettized, unlike every MP4 | its recall looks great for the wrong reason | report as its own confounded tier, never averaged in silently (§4.2) |
| **Reals are single-domain (GOT-10k)** | "real" means GOT-10k; demo clips score fake | diversified reference pool (60/20/20 GOT-10k / YouTube-VOS2 / wild) + wild-real FPR reported in K4 |
| **Generator imbalance in training** (Moonvalley 1 200 vs VideoPoet 100) | kernel specialises on the majority generator | generator-balanced batch sampling in K3 |
| **Inconsistent frame spacing between splits or in the demo** | Δx scale mismatch → MMD meaningless | one canonical sampler everywhere (§4.3b), asserted in the parity test |
| **Watermarks present in test videos, cropped in training frames** | detector keys on the banner; reference set mismatched | apply the same crop at test time; visual contact-sheet check at M3 |
| Drive folder holds fewer GOT-10k reals than 5 000 | split cannot be built as specified | count at M1 **before** extraction; if short, scale both classes down together to keep 1:1 balance |
| Reference set doesn't cover the user's video domain | false positives on real videos | mixed YouTube-VOS2 + GOT-10k pool (Table 9); "wild clip" check in K4; optionally extend the pool later |
| Kaggle 12 h cut-off / session crash | lost GPU hours | 500-video shards + ledger + per-epoch training state (§5.3) |
| `/kaggle/working` 20 GB cap | job dies late in a run | upload-then-delete each shard; never accumulate |
| HF repo gated/renamed, or the authors' Drive folder disappears | pipeline stalls; results not reproducible | pin the **revision SHA**; copy both HF subset and test videos into our own Drive at M3 |
| Sora subset is only 48 videos | per-generator metric is noisy | report it with a confidence interval and say n=48; never headline it alone |
| Vendored code licensing | — | `Video/vendor/NOTICE.md` with guided-diffusion (MIT) and NSG-VD attribution |
| Drive OAuth token expiry | pipeline stalls mid-week | rclone refresh token is long-lived; re-paste `RCLONE_CONF` secret if `rclone lsd` fails in K0 |

---

## 10. Milestones

| # | Milestone | Where | Est. |
|---|---|---|---|
| M1 | Copy the authors' Drive folder into our Drive; **inventory it** (per-generator + GOT-10k real counts) and confirm the 10 000-video split is buildable; pin the HF revision; rclone + Kaggle secret set up | local | ~1 h + copy time |
| M2 | `Video/` skeleton, vendored guided-diffusion, `nsg_features.py` + canonical sampler, unit tests (gate 1) | local | 1 day |
| M3 | K0/K1: dataset pulled + mirrored, manifest built, clips cached, compression audit + contact sheets, gates 2–3 | Kaggle | 2–3 h |
| M4 | K2: NSG features for all 10 000 clips at res 256 (+ augmented training copies, + optional res-128) | Kaggle | 5–6 GPU-h |
| M5 | K3/K4: kernel trained, reference pool + thresholds calibrated | Kaggle | 2–3 GPU-h |
| M6 | K5: per-generator metrics (T2V vs I2V), AIGVDet comparison, local bundle exported (gates 5–7) | Kaggle | 1–2 h |
| M7 | Local `detect.py` + report; parity and end-to-end gates 4, 8 pass | local | 1–2 days |
| M8 | App tab + `Video/README.md` (with the academic-use licence note) + ARCHITECTURE.md section (gate 9) | local | 1 day |

---

## Appendix A — Fixed hyperparameters

| Parameter | Value | Source |
|---|---|---|
| Frames per clip `T` | 8, uniform over the first `min(N, 95)` frames; <8 frames → dropped | NSG-VD C.1 + AIGVDet's 95-frame clip convention (§4.3b) |
| `FRAME_WINDOW` | 95 | AIGVDet preprocessing |
| Clips per video | 1 (all splits); demo uses multi-window aggregation instead | §4.3 / §7.4 |
| Split sizes | train 3 000 R + 3 500 F · ref 500 R · val 500/500 · test 1 000/1 000 (10 000 total) | §4.2 |
| Spatial policy | watermark crop (**all** videos, both classes) → center-crop square → resize | §4.5 |
| Score model | `256x256_diffusion_uncond.pt`, frozen | paper C.3 |
| Diffusion timestep | 5/1000 (`t_value = (steps+1)/1000`) | paper C.3 / repo |
| Score channels | first 3 of 6 (`learn_sigma`) | repo |
| Δx | central difference, forward/backward at boundaries | repo |
| `λ` (NSG denominator) | 1e-10 | paper Alg. 1 |
| Kernel feature dim | 300 | paper C.3 |
| `σ_φ`, `σ_Φ` | 0.1, 100 | paper C.3 |
| Optimizer | Adam, lr 1e-4, wd 0.1, batch 24 | paper C.3 |
| Reference set `n` | 100 | paper E.4 |
| Decision threshold τ | 1.0 default; calibrated at 5 % FPR | paper 4.4 / project convention |

## Appendix B — Commands

```bash
# Kaggle cell 0 (every notebook)
!pip install -q einops timm blobfile decord huggingface_hub imageio
# paste k0_setup_drive.py  → restores rclone.conf, pulls xmulz/AIGVDet + test videos, builds the manifest

# Local: pull artifacts
rclone copy gdrive:SPL3/Video-NSGVD/export       Video/models --progress
rclone copy gdrive:SPL3/Video-NSGVD/results      Video/results --progress

# Local: detect (the demo path)
venv312\Scripts\python -m Video.detect --video test\clip.mp4 --windows 3 --plot
venv312\Scripts\python -m Video.detect --video test\clip.mp4 --fast     # res-128 bundle, ~20-45 s

# Local: app
venv312\Scripts\python -m app
```

## Appendix C — New dependencies

`Video/requirements-video.txt`: `einops`, `timm` (Swin block), `blobfile` (guided-diffusion), `decord`
(fast decode; PyAV/cv2 fallback), `imageio` (GIF — Hotshot clips are GIFs). `torch 2.13.0+cpu`,
`torchvision 0.28.0`, `opencv-python 4.10.0.84` and `imageio 2.37.4` are already installed in `venv312`;
only `einops`, `timm`, `blobfile` and (optionally) `decord` are new locally.

## Appendix D — Citations required

Both papers must be cited in `Video/README.md` and any write-up, and AIGVDet's **academic-research-only**
restriction respected:
- Zhang et al., *Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection*, NeurIPS 2025
  (the method).
- Bai, Lin, Cao, Lou, *AI-Generated Video Detection via Spatial-Temporal Anomaly Learning*, PRCV 2024
  (the AIGVDet/GVD dataset).
- OpenAI guided-diffusion (the frozen score model, MIT licence) — attribution in `Video/vendor/NOTICE.md`.
