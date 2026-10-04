# NSG video detector — Kaggle work summary and local integration guide

**Audience:** the developer and the Claude Code agent integrating the video-detection module into the local DeepTrace application.
**Scope:** everything was trained on Kaggle; the local device only runs **inference (CPU)**. Nothing in this guide requires a GPU or retraining.

---

## 0. Summary

| Item | Value |
|---|---|
| What to integrate | `Detector` class in `detect.py` → `Detector().analyze(video_path) -> dict` |
| Model | Faithful **NSG features** (NeurIPS 2025 NSG-VD) + a **gradient-boosting classifier** on 62 per-clip NSG statistics |
| Not the paper's MMD head | The paper's deep-kernel MMD head was reproduced faithfully but failed to learn on this data (val AUROC 0.554) — see §3 |
| Validation AUROC | **0.969** (generators seen in training) |
| Test AUROC | **0.838** clean / **0.840** JPEG re-compressed (generators **unseen** in training) |
| Decision threshold | `p_fake > 0.6915` → "AI-generated" (calibrated for 5% false positives on validation reals; 4.8% on unseen test reals) |
| Files needed locally | code zip, UNet weights (2.1 GB), detector `.joblib`, parity file — §5 |
| Hard version pins | `scikit-learn==1.6.1`, `numpy` 2.x (the detector was pickled with these) |
| Local cost | ~2.2 GB RAM for the network; very roughly 0.5–1.5 min per video on a laptop CPU (estimate — measure) |

> `PLAN.md`'s local section assumed the MMD bundle (`nsgvd_local_res224.pt` + reference set). **That is superseded.** Use the gradient-boosting detector described here.

---

## 1. Background

- **Paper:** Zhang et al., *Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection*, NeurIPS 2025 (arXiv:2510.08073).
- **Official code:** `github.com/ZSHsh98/NSG-VD` @ `9a60f57fcb464b418899110a4717c8c8fd7687bc`.
- **Idea:** the *Normalized Spatiotemporal Gradient* NSG = ∇ₓ log p / (−∂ₜ log p + λ), estimated with a pre-trained diffusion model's score function (spatial part) and a brightness-constancy temporal derivative (temporal part).
- **Score network:** OpenAI guided-diffusion `256x256_diffusion_uncond.pt` (552 M parameters, frozen), code pinned at `openai/guided-diffusion` @ `22e0df8183507e13a7813f8d38d51b072ca1e67c`.

### Implementation facts verified against the official code

These were tested numerically in a sandbox against the cloned official repository:

| Component | Result |
|---|---|
| NSG computation (score, central difference, denominator, NSG) | **bit-identical** to official `_get_score_features` / `_get_score_velocity` at 224 and 128 resolution |
| Deep kernel (TALL Swin + `deep_MMD`) | official checkpoint `standard-SEINE-mp.pth` loads with `strict=True`; features, MMD-MP loss, float32 and float64 gradients and the test statistic all **identical** |
| Score definition | `score = −UNet(2x−1, t·999)[:, :3] / σ_VPSDE(t)`, `t = 0.006`, VP-SDE β ∈ [0.1, 20], evaluated on the **clean** frame (the official code adds **no** noise) |
| Official details not in the paper | resolution 224 everywhere; NSG of the first and last frame multiplied by 100 at load time; frames resized with PIL LANCZOS |

---

## 2. What was done on Kaggle

### 2.1 Environment (Kaggle, 2× Tesla T4)

Python 3.12.13 · torch 2.10.0+cu128 · numpy 2.0.2 · pandas 2.3.3 · scikit-learn 1.6.1 · opencv 4.13.0 · Pillow 11.3.0 · matplotlib 3.10.0 · rclone v1.74.4 (SHA-256 `fe435e0c…`) · NCCL for multi-GPU.
Google Drive access via **rclone with the user's own OAuth client** (Kaggle Secret `RCLONE_CONF`, base64 of `rclone.conf`). All stages were resumable batch-by-batch with dual checkpoints (Kaggle local storage + Google Drive); see §4 for the Drive layout.

### 2.2 Data

| Class | Source | Videos on Drive | Notes |
|---|---|---|---|
| Fake | AIGVDet **GVD** test set (authors' shared Drive folder, 11 zip archives, 15.3 GB), unpacked to `videos_subset/T2V/*_mp4` and `videos_subset/I2V/*_mp4` | 11,068 | T2V: Moonvalley 3000, VideoCraft 1500, Pika 1000, NeverEnds 1000, Emu 900, Hotshot 500, VideoPoet 120, Sora 48 · I2V: Moonvalley 1000, Pika 1000, NeverEnds 1000 |
| Real | **Kinetics-400** (Kaggle dataset `ipythonx/k4testset`, `videos_val`) | 3,000 | copied to `videos_subset/real/kinetics400/` |
| Real | **MSR-VTT** (Kaggle dataset `vishnutheepb/msrvtt`, `TrainValVideo`) | 2,000 | copied to `videos_subset/real/msrvtt/` |

The GVD folder contained **no real videos**, so reals follow the NSG-VD paper's own protocol (Kinetics-400 for training, MSR-VTT for testing). Three files are identical between `T2V/neverends_mp4` and `I2V/neverends_mp4`; I2V NeverEnds and I2V Pika are not used, so this cannot leak.

**Splits** (`manifests/manifest_v1.csv`, seed 42, frozen):

| Split | Real | Fake |
|---|---|---|
| train | 3000 (Kinetics 2228, MSR-VTT 772) | 3500 (Moonvalley T2V 1050, VideoCraft 700, Pika 613, NeverEnds 612, Emu 437, VideoPoet 88) |
| val | 500 (Kinetics 390, MSR-VTT 110) | 500 (same six generators, proportional) |
| ref | 500 (Kinetics 382, MSR-VTT 118) | — |
| test | 1000 (MSR-VTT) | 1000 (Hotshot 500, Moonvalley I2V 452, Sora 48) — all **unseen** generators or modality |

### 2.3 Pipeline stages

| Stage | What it did | Output on Drive |
|---|---|---|
| K0 | Environment checks, rclone, inventory, deterministic manifest | `manifests/` |
| K1 | Decode → **canonical 8 frames** per video (CPU, 100 batches of 100, 0 clips skipped) | `frames/r224/<split>/shard_*.npy` (uint8, 8×3×224×224) |
| K2 | **NSG features** on 2× T4 (fp16 UNet), variants: train clean/jpeg/rescale, test clean/jpeg, ref/val clean — 250 batches; sanity gate passed (denoising improved ≈4×, ε-cosine 0.87, fp16-vs-fp32 NSG cosine 0.9994, no ∂ₜlog p values in [−0.1, 0.1], matching paper Fig. 5) | `nsg/steps5_res224/<variant>/<split>/shard_*.npy` (stored as **score fp16 + denominator in the JSON marker**) |
| K3 | Paper-faithful **deep-kernel MMD-MP** training (DDP, official hyper-parameters) | `ckpts/res224_mp_gvd6/` |
| K4–K5 | Reference set, threshold calibration, test evaluation, authors' checkpoint baseline | `reference/`, `results/res224_mp_gvd6/` |
| Diagnostics | Linear probes on NSG statistics; official whole-video sampling experiment (ref/val only) | `frames/r224_full/`, `nsg/steps5_res224_full/` |
| **Final** | **NSG statistics + gradient boosting** trained on the train split (6500 clips × 3 variants = 19,500 rows) | `export/nsg_stats_gbm.joblib` |

### 2.4 Canonical preprocessing (identical on Kaggle and locally)

1. Decode up to the first **95** frames (OpenCV; rotation metadata honoured; GIF via Pillow).
2. Pick **8 frames**: `W = min(N, 95)`, `stride = W // 8`, indices `0, stride, …, 7·stride`. Clips with fewer than 8 decodable frames are rejected.
3. Remove the bottom **10%** band (watermark policy, applied to every video of both classes).
4. Centre **square** crop → resize to **224×224 with PIL LANCZOS** → uint8.

### 2.5 Why the final model is not the paper's MMD head

| Evidence | Value |
|---|---|
| Paper-recipe MMD deep kernel, best val AUROC | 0.554 (epoch 20; early-stopped at 50) |
| Its training statistic | MMD² never became positive (−0.28 → −0.01): the kernel never separated the classes |
| Kernel bandwidth σ₀ | fell linearly 0.097 → 0.0018 — the official weight decay (0.1) drives it to zero |
| Architecture issue | the Swin's per-patch LayerNorm divides out signal **magnitude**, where the NSG signal lives |
| Linear probe on NSG statistics (val, 5-fold) | 0.873 — the features carry the signal; the head did not use it |
| Authors' released checkpoint on these features | 0.53 (windowed sampling, test) · 0.678 (official whole-video sampling, val) |

A supervised head on NSG features is the paper's own **NSG-BCE** variant idea (Appendix E.3). Our version uses 62 interpretable NSG statistics and gradient boosting.

---

## 3. Results of the final detector

**Validation** (generators seen in training): AUROC **0.969**, mean per generator 0.959 — Emu 0.998, Moonvalley T2V 0.983, VideoCraft 0.969, Pika 0.956, NeverEnds 0.947, VideoPoet 0.904.

**Test** (unseen generators; threshold τ = 0.6915 from validation reals):

| Generator (tier) | n | AUROC clean (95% CI) | Recall @ τ | AUROC JPEG |
|---|---|---|---|---|
| Moonvalley I2V (easiest: seen family, unseen modality) | 452 | 0.912 (0.896–0.928) | 68.6% | 0.914 |
| Sora (hardest: near-photorealistic) | 48 | 0.878 (0.825–0.926) | 58.3% | 0.710 |
| Hotshot (short, low-resolution clips) | 500 | 0.768 (0.743–0.791) | 14.2% | 0.786 |
| **All** | 1000 | **0.838** (0.821–0.856) | 40.9% | **0.840** |

False-positive rate on the 1000 unseen test reals at τ: **4.8%** (target 5%), i.e. the calibration transfers.

### Diagnostic probes (val, 5-fold logistic regression)

| Probe | Windowed sampling (used) | Official whole-video sampling |
|---|---|---|
| Temporal part of NSG (denominators) | 0.563 | 0.616 |
| Spatial part of NSG (score energy) | 0.846 | 0.849 |
| Both NSG parts | 0.873 | 0.898 |
| **File metadata only (fps, frames, resolution, size)** | **1.000** | **1.000** |

### Known limitations (state these in the report and the UI)

- **Shortcut risk:** file metadata alone separates real and fake perfectly in this dataset (all reals ≤ 568×320 at ~30 fps and long; fakes mostly ≥ 512 px, 8–24 fps, often short; exact values in `results/encoding_audit.csv`). Part of the detector's accuracy likely comes from resolution/compression differences.
- **Hotshot** (tiny, heavily compressed clips) is caught only 14% of the time — consistent with that shortcut.
- **Sora** drops from 0.878 to 0.710 after JPEG re-compression — reliance on fine high-frequency detail.
- **Wild-video check still to do:** run 10–20 high-resolution real phone videos locally. If many are flagged "AI-generated", show this limitation in the app.

---

## 4. Google Drive inventory — `gdrive:SPL3/Video-NSGVD/`

| Path | Content | Needed locally? |
|---|---|---|
| `export/nsg_stats_gbm.joblib` | **final detector** (dict: `model`, `features` [62 names, order matters], `tau_fpr05` = 0.6915, `feature_dir` = `nsg/steps5_res224`) | **yes** |
| `export/nsgvd_local.zip` | local inference code (created by Kaggle Cell C, Appendix A) | **yes** |
| `export/parity_samples.npz` | 5 test clips: frames + Kaggle's stored score/denominator (Kaggle Cell D) | **yes (verification)** |
| `models/256x256_diffusion_uncond.pt` (+ `.sha256`) | guided-diffusion UNet weights, 2.1 GB, SHA-256 starts `a37c32fffd31` | **yes** |
| `export/nsgvd_local_res224.pt`, `export/SHA256SUMS.txt` | old MMD bundle (val 0.554) | **no — do not use** |
| `videos_subset/` | 16,068 source videos (18.5 GB): `T2V/`, `I2V/`, `real/`, `_unpacked/` markers | no |
| `TestVideos` | shortcut to the AIGVDet authors' folder (zip archives) | no |
| `manifests/manifest_v1.csv`, `manifest_spec.json` | frozen splits | no |
| `frames/r224/…` | canonical frames, windowed sampling | no |
| `nsg/steps5_res224/…` | NSG features (score fp16 `.npy` + JSON markers with denominators) | no |
| `frames/r224_full/…`, `nsg/steps5_res224_full/…` | official-sampling experiment (ref/val) | no |
| `ckpts/res224_mp_gvd6/…` | failed MMD run checkpoints + `history.json` | no |
| `reference/`, `results/` | K4/K5 outputs, sanity report, encoding audit, contact sheet, metrics CSVs | no (report material) |
| `ledgers/` | per-session processing logs | no |

The 62-feature cache used to train the detector lives only in Kaggle's `/kaggle/working/nsgvd_ckpt/results/nsg_stats/` (not on Drive).

---

## 5. Files for the local device

### 5.1 Target layout

```
nsgvd-local/
├── detect.py                         # entry point + Detector class (Appendix B2)
├── requirements.txt
├── README.txt
├── nsgvd/
│   ├── __init__.py
│   ├── config.py                     # constants (paths are env-overridable)
│   ├── utils.py
│   ├── video.py                      # canonical 8-frame extraction
│   ├── nsg.py                        # UNet loading + score / NSG denominator
│   └── stats.py                      # 62 NSG statistics (Appendix B1)
├── third_party/guided-diffusion/guided_diffusion/*.py   # pinned @ 22e0df8 (MIT)
└── models/
    ├── 256x256_diffusion_uncond.pt   # 2.1 GB
    ├── nsg_stats_gbm.joblib
    └── parity_samples.npz
```

Not needed locally: `kernel.py`, `drive.py`, `store.py`, `manifest.py`, `data.py`, `evaluate.py`, `env.py`, the K1–K3 stage scripts.

### 5.2 Getting the files

**Prerequisite on Kaggle (one time, CPU session is enough):**
1. In notebook Cell 2, restore the main settings:
   `FRAME_WINDOW = 95`, `D_FRAMES = f"frames/r{FEATURE_SIZE}"`, `D_NSG = f"nsg/steps{DIFFUSE_STEPS}_res{SCORE_RESOLUTION}"`, `SPLITS_TO_PROCESS = ["ref", "val", "train", "test", "wild"]`.
2. Run Cells 1–17, then Cells A–D from **Appendix A**. Cell C uploads `export/nsgvd_local.zip`; Cell D uploads `export/parity_samples.npz`.

**On the local PC (PowerShell):**

```powershell
$RC = "D:\rclone-v1.75.1-windows-amd64\rclone-v1.75.1-windows-amd64\rclone.exe"
New-Item -ItemType Directory -Force D:\nsgvd-local | Out-Null
cd D:\nsgvd-local
& $RC copy gdrive:SPL3/Video-NSGVD/export/nsgvd_local.zip . -P
Expand-Archive .\nsgvd_local.zip -DestinationPath . -Force
& $RC copy gdrive:SPL3/Video-NSGVD/models/256x256_diffusion_uncond.pt .\models -P
& $RC copy gdrive:SPL3/Video-NSGVD/export/nsg_stats_gbm.joblib .\models -P
& $RC copy gdrive:SPL3/Video-NSGVD/export/parity_samples.npz .\models -P
(Get-FileHash .\models\256x256_diffusion_uncond.pt -Algorithm SHA256).Hash.ToLower()   # must start with a37c32fffd31
```

If the zip is unavailable, use the fallback in **Appendix D**.

---

## 6. Local environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` (inside the zip):

```
numpy>=2.0,<2.3
scikit-learn==1.6.1
pandas>=2.0
joblib>=1.3
opencv-python>=4.8
pillow>=10
```

- **`scikit-learn==1.6.1` and numpy 2.x are mandatory:** the detector is a pickled `HistGradientBoostingClassifier`. Other versions may refuse to load it or load it incorrectly. To change versions, re-export the model on Kaggle.
- torch: any recent CPU build (≥ 2.1). No GPU needed; CUDA is used automatically if present.
- If the app already has its own environment, add these pins there instead of creating a second venv. Resolve any conflict in favour of the two mandatory pins.

---

## 7. Inference contract — do not change

| Step | Exact behaviour | Where |
|---|---|---|
| Frame sampling | first ≤ 95 frames, stride `W // 8`, 8 frames from frame 0. `detect.py` forces `FRAME_WINDOW = 95` from the bundle's `feature_dir`, even if `config.py` says otherwise | `video.py`, `detect.py` |
| Spatial | bottom 10% removed → centre square → 224×224 PIL LANCZOS | `video.py` |
| Score | guided-diffusion UNet, input `2x−1`, timestep label `0.006·999`, `score = −ε / σ_VPSDE(0.006)` | `nsg.py` |
| Denominator | `⟨score_t, Δx_t⟩ + 1e-10`, Δx = central difference (one-sided at ends), x in [0, 1] | `nsg.py` |
| Precision | score rounded through **fp16** before statistics (Kaggle stored fp16); local UNet runs fp32 on CPU | `detect.py` |
| Features | `clip_stats(score, denom)` → 62 values. Columns are selected **by name in `bundle["features"]` order** | `stats.py`, `detect.py` |
| Decision | `p_fake = model.predict_proba(x)[0, 1]`; "AI-generated" if `p_fake > bundle["tau_fpr05"]` (0.6915) | `detect.py` |

**`Detector.analyze()` output:**

```json
{"file": "clip.mp4", "p_fake": 0.8731, "threshold": 0.6915, "verdict": "AI-generated",
 "seconds": 41.2, "frames_used": [0, 11, 22, 33, 44, 55, 66, 77], "fps": 30.0, "resolution": "1920x1080"}
```

Errors: `nsgvd.video.VideoDecodeError` for unreadable files or clips with fewer than 8 frames; `FileNotFoundError` if model files are missing.

---

## 8. Integration guide for the existing application

Instructions for the Claude Code agent. Adapt names and structure to the existing repository's conventions.

1. **Placement:** put `nsgvd-local/` inside the existing video module (for example `Video/nsgvd_local/`). Keep the inner package name **`nsgvd`**, because the modules import each other as `nsgvd.*`. `detect.py` adds its own folder to `sys.path`; if you import it from elsewhere, import `Detector` from that file or put the folder on the path once at startup.
2. **Model files:** keep the 2.1 GB weights out of git. Load paths from the app's config or environment if it has a settings system; otherwise keep `models/` next to `detect.py` (the defaults).
3. **Load once:** create a single `Detector()` at application start (singleton or a lazily initialised module-level instance). Loading the UNet takes ~20–40 s and ~2.2 GB RAM.
4. **Run off the UI thread:** `analyze()` is CPU-heavy (roughly 0.5–1.5 min per video, estimate). Run it in a worker thread or process with **concurrency 1**, a progress or "processing" state, and a generous timeout. On low-memory machines, lower `micro_batch` in `Detector.__init__` from 8 to 4.
5. **Wrap, don't modify:** expose an adapter matching the app's existing detector interface (e.g. `VideoNSGDetector.predict(path) -> result`), mapping `p_fake`/`verdict` to the app's common result schema. Do not change preprocessing, features, threshold or model inside the adapter.
6. **Presentation:** show `p_fake` and the verdict. Recommended UI policy (presentation only): an **"uncertain"** label when `|p_fake − 0.6915| < 0.10`, plus a short note on the known limitations (§3).
7. **Errors:** catch `VideoDecodeError` and return a clear message ("video too short or unreadable"). Log `seconds`, `resolution` and `fps` for each analysis.
8. **Tests:** add (a) a parity test that runs `Detector().parity("models/parity_samples.npz")` and asserts the returned difference < 0.05, and (b) a smoke test on one short sample clip that asserts the output keys and a probability within [0, 1]. Inference is deterministic (no randomness).

---

## 9. Acceptance checklist

- [ ] UNet SHA-256 starts with `a37c32fffd31`
- [ ] `python detect.py --parity models\parity_samples.npz` prints **`PARITY OK`** (differences come from fp32 CPU vs fp16 GPU and should be small)
- [ ] `python detect.py some_video.mp4` prints a verdict; record the time per video on this PC
- [ ] the app shows verdict + probability without freezing the UI
- [ ] wild check: 10–20 high-resolution real phone videos; record how many are flagged (expected limitation)
- [ ] `rclone.conf`, tokens and model weights are **not** committed to git

---

## 10. Do not

- retrain or fine-tune locally, or change any constant in §7 — features would no longer match the trained model;
- use `export/nsgvd_local_res224.pt` (failed MMD model) or the `_full` feature folders for this detector;
- upgrade scikit-learn or downgrade numpy below 2.0 without re-exporting the model on Kaggle;
- describe the detector as "the paper's MMD method" or as pure physics — it is NSG features + a supervised head, with the caveats in §3;
- commit secrets (`rclone.conf`, OAuth tokens). If they are ever exposed, revoke them and create new ones.

---

## 11. Reproducing or retraining (Kaggle)

- The notebook (`nsg-implementation.ipynb`) contains every module as a `%%writefile` cell plus the run cells. Every stage resumes from Google Drive.
- The detector was produced by the last notebook cell (**Appendix C**): it reads `nsg/steps5_res224`, caches per-shard statistics, trains `HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)` on train (clean + jpeg + rescale), calibrates τ on val reals at 5% FPR, evaluates test, and saves `export/nsg_stats_gbm.joblib`.
- After retraining, re-run Cells C and D and re-download both files.
- Promising improvements: add higher-resolution real videos so that "high resolution" no longer implies "fake" (likely helps Hotshot and the shortcut); retrain on official whole-video sampling features (val probe 0.898 vs 0.873).

---

## Appendix A — Kaggle cells that produce the local files

**Cell A** — write `stats.py`: a cell whose first line is
`%%writefile /kaggle/working/nsgvd_code/nsgvd/stats.py`
followed by the exact contents of Appendix B1.

**Cell B** — write `detect.py`: a cell whose first line is
`%%writefile /kaggle/working/nsgvd_code/detect.py`
followed by the exact contents of Appendix B2.

**Cell C** — check every file for truncation, zip, and upload `export/nsgvd_local.zip` (uses the notebook's `drive` object from Cell 17):

```python
import py_compile, zipfile
from nsgvd import config as C

LAST_LINE = {"__init__.py": '__version__ = "1.0.0"', "config.py": 'SPLITS_TO_PROCESS = [',
             "utils.py": 'return f"{seconds // 3600}h', "video.py": "return _to_tchw(out)",
             "nsg.py": "if values.numel() else math.nan", "stats.py": "return f", "detect.py": "main()"}
files = {f"nsgvd/{n}": C.CODE_ROOT / "nsgvd" / n for n in LAST_LINE if n != "detect.py"}
files["detect.py"] = C.CODE_ROOT / "detect.py"
for path in sorted((C.GUIDED_DIFFUSION_DIR / "guided_diffusion").glob("*.py")):
    files[f"third_party/guided-diffusion/guided_diffusion/{path.name}"] = path
if (C.GUIDED_DIFFUSION_DIR / "LICENSE").exists():
    files["third_party/guided-diffusion/LICENSE"] = C.GUIDED_DIFFUSION_DIR / "LICENSE"

problems = []
for arc, path in files.items():
    if not path.exists():
        problems.append(f"missing: {path}")
        continue
    if path.suffix == ".py":
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            problems.append(f"syntax error: {arc}: {exc.exc_value}")
        name = path.name
        if arc.startswith(("nsgvd/", "detect")) and name in LAST_LINE:
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines or LAST_LINE[name] not in lines[-1]:
                problems.append(f"truncated: {arc} (re-paste its cell)")
if problems:
    raise RuntimeError("Fix these before packaging:\n  " + "\n  ".join(problems))

requirements = ("# torch: install separately ->  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "numpy>=2.0,<2.3\nscikit-learn==1.6.1\npandas>=2.0\njoblib>=1.3\nopencv-python>=4.8\npillow>=10\n")
readme = ("NSG detector - local inference\n"
          "1. put 256x256_diffusion_uncond.pt and nsg_stats_gbm.joblib (+ parity_samples.npz) into models/\n"
          "2. pip install torch --index-url https://download.pytorch.org/whl/cpu ; pip install -r requirements.txt\n"
          "3. python detect.py --parity models/parity_samples.npz\n"
          "4. python detect.py my_video.mp4\n")
out = C.SCRATCH_DIR / "nsgvd_local.zip"
out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for arc, path in files.items():
        zf.write(path, arc)
    zf.writestr("requirements.txt", requirements)
    zf.writestr("README.txt", readme)
    zf.writestr("models/PUT_MODEL_FILES_HERE.txt", "256x256_diffusion_uncond.pt, nsg_stats_gbm.joblib, parity_samples.npz\n")
drive.upload(out, f"{C.D_EXPORT}/{out.name}")
print(f"packaged {len(files)} files ({out.stat().st_size / 1e3:.0f} KB) -> Google Drive {C.D_EXPORT}/{out.name}")
```

**Cell D** — build `export/parity_samples.npz` from the same windowed test features the detector was evaluated on:

```python
import numpy as np
from nsgvd import config as C
from nsgvd.store import CheckpointStore

NSG_DIR, FRAMES_DIR = "nsg/steps5_res224", "frames/r224"          # windowed features the detector was trained on
store = CheckpointStore("parity", drive, async_upload=False)
nsg = store.scan(f"{NSG_DIR}/clean/test", manage=False)
unit = sorted(nsg)[0]
marker = nsg[unit].marker
frames_unit = marker["frames_unit"]                                # K1 batch these features were computed from
fmarker = store.scan(frames_unit.rsplit("/", 1)[0], manage=False)[frames_unit].marker
fpath, ftmp = store.fetch(frames_unit, fmarker, C.SCRATCH_DIR / "parity")
spath, stmp = store.fetch(unit, marker, C.SCRATCH_DIR / "parity")
f_ids = [it["clip_id"] for it in fmarker["items"]]
s_ids = [it["clip_id"] for it in marker["items"]]
pick = s_ids[:5]
frames, score = np.load(fpath, mmap_mode="r"), np.load(spath, mmap_mode="r")
out = C.SCRATCH_DIR / "parity_samples.npz"
np.savez_compressed(out, clip_ids=np.array(pick),
                    frames=np.stack([frames[f_ids.index(c)] for c in pick]),
                    score=np.stack([score[s_ids.index(c)] for c in pick]),
                    denom=np.array([marker["denom"][s_ids.index(c)] for c in pick], dtype=np.float32))
drive.upload(out, f"{C.D_EXPORT}/parity_samples.npz")
print(f"parity file with {len(pick)} test clips from {unit} -> Google Drive {C.D_EXPORT}/parity_samples.npz")
```

---

## Appendix B1 — `nsgvd/stats.py`

```python
"""Per-clip NSG statistics used by the gradient-boosting detector.

Must stay byte-for-byte identical in behaviour to the function the detector was trained with
(notebook cell "NSG-statistics detector"), otherwise the classifier sees different features.
"""
from __future__ import annotations

import numpy as np


def clip_stats(s: np.ndarray, d: np.ndarray) -> dict:
    """s: (T,3,H,W) score maps (float32), d: (T,) NSG denominators -> scale-aware NSG statistics."""
    a = np.abs(s)
    e = a.mean(axis=(1, 2, 3))                                   # score energy per frame
    ech = a.mean(axis=(2, 3))                                    # ... per colour channel
    hf = (np.abs(np.diff(s, axis=-1)).mean(axis=(1, 2, 3)) +
          np.abs(np.diff(s, axis=-2)).mean(axis=(1, 2, 3))) / (2 * e + 1e-12)   # spatial roughness
    f = {}
    for t in range(s.shape[0]):
        f[f"logden{t}"] = np.log10(abs(d[t]) + 1e-12)
        f[f"sign{t}"] = float(np.sign(d[t]))
        f[f"logE{t}"] = np.log10(e[t] + 1e-12)
        f[f"hf{t}"] = float(hf[t])
        for c in range(s.shape[1]):
            f[f"logE{t}c{c}"] = np.log10(ech[t, c] + 1e-12)
    for k in ("logden", "logE", "hf"):
        v = np.array([f[f"{k}{t}"] for t in range(s.shape[0])])
        f[f"{k}_mean"], f[f"{k}_std"] = float(v.mean()), float(v.std())
    return f
```

## Appendix B2 — `detect.py`

```python
"""Local NSG detector: video -> 8 canonical frames -> NSG (guided-diffusion score) -> statistics -> classifier.

    python detect.py VIDEO [VIDEO ...] [--json]
    python detect.py --parity models/parity_samples.npz     # check this PC reproduces Kaggle's features

Expected layout next to this file:
    nsgvd/                                 __init__ config utils video nsg stats (.py)
    third_party/guided-diffusion/          openai/guided-diffusion @ 22e0df8 (pinned)
    models/256x256_diffusion_uncond.pt     UNet weights      (Drive: models/)
    models/nsg_stats_gbm.joblib            trained detector  (Drive: export/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NSGVD_WORK", str(ROOT / "work"))
os.environ.setdefault("NSGVD_SCRATCH", str(ROOT / "work" / "scratch"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nsgvd import config as C  # noqa: E402

C.GUIDED_DIFFUSION_DIR = ROOT / "third_party" / "guided-diffusion"
UNET_PATH = ROOT / "models" / C.UNET_FILENAME
DETECTOR_PATH = ROOT / "models" / "nsg_stats_gbm.joblib"
# frame sampling the detector was trained with (features recorded in the bundle's "feature_dir")
TRAINED_FRAME_WINDOW = {"nsg/steps5_res224": 95, "nsg/steps5_res224_full": 100_000}

from nsgvd.nsg import NSGEstimator, load_unet  # noqa: E402
from nsgvd.stats import clip_stats  # noqa: E402
from nsgvd.video import read_clip  # noqa: E402


class Detector:
    """Loads the score network + classifier once; `analyze(path)` scores one video."""

    def __init__(self, device: str | None = None):
        for path in (UNET_PATH, DETECTOR_PATH):
            if not path.exists():
                raise FileNotFoundError(f"missing {path} -- download it from Google Drive (see README)")
        self.bundle = joblib.load(DETECTOR_PATH)
        C.FRAME_WINDOW = TRAINED_FRAME_WINDOW.get(self.bundle.get("feature_dir"), C.FRAME_WINDOW)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        self.est = NSGEstimator(load_unet(UNET_PATH, self.device), self.device, micro_batch=8)

    def features(self, frames: np.ndarray) -> dict:
        """(T, 3, 224, 224) uint8 canonical frames -> the detector's NSG statistics."""
        score, denom = self.est.score_and_denom(torch.from_numpy(np.ascontiguousarray(frames)[None]))
        s = score[0].half().float().cpu().numpy()          # same fp16 rounding as the stored training features
        return clip_stats(s, denom[0].double().cpu().numpy())

    def predict(self, stats: dict) -> float:
        x = pd.DataFrame([stats])[self.bundle["features"]]
        return float(self.bundle["model"].predict_proba(x)[0, 1])

    def analyze(self, path: str | Path) -> dict:
        t0 = time.time()
        frames, info = read_clip(path)
        p_fake = self.predict(self.features(frames))
        tau = float(self.bundle["tau_fpr05"])
        return {"file": str(path), "p_fake": round(p_fake, 4), "threshold": round(tau, 4),
                "verdict": "AI-generated" if p_fake > tau else "real", "seconds": round(time.time() - t0, 1),
                "frames_used": info["indices"], "fps": info["fps"], "resolution": f"{info['width']}x{info['height']}"}

    def parity(self, npz_path: str | Path) -> float:
        """Recompute 5 Kaggle test clips locally; returns the largest probability difference."""
        data = np.load(npz_path)
        worst = 0.0
        for cid, frames, s_ref, d_ref in zip(data["clip_ids"], data["frames"], data["score"], data["denom"]):
            p_local = self.predict(self.features(frames))
            p_kaggle = self.predict(clip_stats(s_ref.astype(np.float32), d_ref.astype(np.float64)))
            worst = max(worst, abs(p_local - p_kaggle))
            print(f"{str(cid)[:44]:44s} p_fake local={p_local:.4f} kaggle={p_kaggle:.4f} |diff|={abs(p_local - p_kaggle):.4f}")
        print(f"PARITY {'OK' if worst < 0.05 else 'CHECK'}: largest difference {worst:.4f} (fp32 CPU vs fp16 GPU)")
        return worst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="*", help="video files (.mp4, .mov, .webm, .gif, ...)")
    parser.add_argument("--parity", help="parity_samples.npz from Drive export/")
    parser.add_argument("--json", action="store_true", help="print one JSON object per video")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    args = parser.parse_args(argv)
    if not args.videos and not args.parity:
        parser.error("give at least one video, or --parity FILE")
    detector = Detector(args.device)
    if args.parity:
        detector.parity(args.parity)
    for video in args.videos:
        result = detector.analyze(video)
        if args.json:
            print(json.dumps(result))
        else:
            print(f"{result['verdict']:13s} p_fake={result['p_fake']:.3f} (threshold {result['threshold']:.3f}) "
                  f"{result['resolution']} {result['seconds']}s  {result['file']}")


if __name__ == "__main__":
    main()
```

---

## Appendix C — training cell that produced `nsg_stats_gbm.joblib`

The last cell of the Kaggle notebook, shown for traceability. It uses the notebook's `drive` and `display`, and its `clip_stats` is identical to Appendix B1.

```python
import joblib
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from nsgvd import config as C
from nsgvd.evaluate import auroc_summary, calibrate_threshold, generator_table
from nsgvd.store import CheckpointStore

FEATURE_DIR = "nsg/steps5_res224"            # windowed NSG features: complete for train / val / test
store = CheckpointStore("nsgstats", drive, async_upload=False)
statuses = store.scan(FEATURE_DIR, manage=False)
cache = C.LOCAL_CKPT_DIR / C.D_RESULTS / "nsg_stats" / FEATURE_DIR.replace("/", "_")
cache.mkdir(parents=True, exist_ok=True)


def clip_stats(s: np.ndarray, d: np.ndarray) -> dict:
    """s: (T,3,H,W) score maps, d: (T,) NSG denominators -> scale-aware NSG statistics."""
    a = np.abs(s)
    e = a.mean(axis=(1, 2, 3))                                   # score energy per frame
    ech = a.mean(axis=(2, 3))                                    # ... per colour channel
    hf = (np.abs(np.diff(s, axis=-1)).mean(axis=(1, 2, 3)) +
          np.abs(np.diff(s, axis=-2)).mean(axis=(1, 2, 3))) / (2 * e + 1e-12)   # spatial roughness
    f = {}
    for t in range(s.shape[0]):
        f[f"logden{t}"] = np.log10(abs(d[t]) + 1e-12)
        f[f"sign{t}"] = float(np.sign(d[t]))
        f[f"logE{t}"] = np.log10(e[t] + 1e-12)
        f[f"hf{t}"] = float(hf[t])
        for c in range(s.shape[1]):
            f[f"logE{t}c{c}"] = np.log10(ech[t, c] + 1e-12)
    for k in ("logden", "logE", "hf"):
        v = np.array([f[f"{k}{t}"] for t in range(s.shape[0])])
        f[f"{k}_mean"], f[f"{k}_std"] = float(v.mean()), float(v.std())
    return f


def cache_file(unit: str) -> Path:
    return cache / (unit.replace("/", "__") + ".csv")


todo = [u for u, st in sorted(statuses.items()) if st.on_drive and not cache_file(u).exists()]
print(f"{len(statuses)} feature shards | {len(statuses) - len(todo)} already summarised | {len(todo)} to process")
for k, unit in enumerate(todo, 1):
    st = statuses[unit]
    path, is_tmp = store.fetch(unit, st.marker, C.SCRATCH_DIR / "nsgstats")
    arr = np.load(path, mmap_mode="r")
    bad = set(st.marker.get("nonfinite", []))
    rows = [{"clip_id": item["clip_id"], "label": int(item["label"]), "source": item["source"],
             "split": st.marker["split"], "variant": st.marker["variant"],
             **clip_stats(np.asarray(arr[i], dtype=np.float32), np.asarray(denom, dtype=np.float64))}
            for i, (item, denom) in enumerate(zip(st.marker["items"], st.marker["denom"])) if item["clip_id"] not in bad]
    tmp_csv = cache_file(unit).with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp_csv, index=False)
    tmp_csv.replace(cache_file(unit))                            # atomic: resumable after a crash
    del arr
    if is_tmp:
        Path(path).unlink(missing_ok=True)
    if k % 10 == 0 or k == len(todo):
        print(f"  {k}/{len(todo)} shards summarised")

feats = pd.concat([pd.read_csv(p) for p in sorted(cache.glob("*.csv"))], ignore_index=True)
cols = [c for c in feats.columns if c not in ("clip_id", "label", "source", "split", "variant")]
train = feats[feats["split"] == "train"]                         # clean + jpeg + rescale copies
val = feats[(feats["split"] == "val") & (feats["variant"] == "clean")].copy()
print(f"train rows {len(train)} ({train['clip_id'].nunique()} clips) | val clips {len(val)} | features {len(cols)}")
clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                     l2_regularization=1.0, random_state=0)
clf.fit(train[cols], train["label"])
val["score"] = clf.predict_proba(val[cols])[:, 1]
res = auroc_summary(val)
print(f"\nVAL  AUROC all = {res['val_auroc_all']:.3f} | mean per generator = {res['val_auroc_mean_gen']:.3f}")
print("     " + "  ".join(f"{g}={a:.3f}" for g, a in res["val_auroc_by_gen"].items()))
tau = calibrate_threshold(val.loc[val["label"] == 0, "score"].to_numpy())
print(f"tau @ 5% FPR on val reals = {tau:.4f}")
for variant in ("clean", "jpeg"):
    part = feats[(feats["split"] == "test") & (feats["variant"] == variant)].copy()
    if part.empty:
        continue
    part["score"] = clf.predict_proba(part[cols])[:, 1]
    print(f"\n=== TEST ({variant}) - NSG statistics + gradient boosting ===")
    display(generator_table(part, tau, 0.5)[["generator", "n_fake", "AUROC", "AUROC_95CI",
                                             "Recall@cal", "FPR@cal", "Acc@cal(bal)"]].round(3))
out = C.LOCAL_CKPT_DIR / C.D_EXPORT / "nsg_stats_gbm.joblib"
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": clf, "features": cols, "tau_fpr05": tau, "feature_dir": FEATURE_DIR}, out)
drive.upload(out, f"{C.D_EXPORT}/{out.name}")
print(f"\nsaved detector -> Google Drive {C.D_EXPORT}/{out.name}")
```

---

## Appendix D — fallback if `nsgvd_local.zip` is not available

1. Extract the module sources from the notebook's `%%writefile` cells (run next to `nsg-implementation.ipynb`):

```python
import json
from pathlib import Path

WANTED = {"nsgvd/__init__.py", "nsgvd/config.py", "nsgvd/utils.py", "nsgvd/video.py",
          "nsgvd/nsg.py", "nsgvd/stats.py", "detect.py"}
nb = json.load(open("nsg-implementation.ipynb", encoding="utf-8"))
for cell in nb["cells"]:
    src = "".join(cell.get("source", [])).lstrip()
    if not src.startswith("%%writefile "):
        continue
    first, body = src.split("\n", 1)
    rel = first.split(maxsplit=1)[1].strip().replace("/kaggle/working/nsgvd_code/", "")
    if rel in WANTED:
        out = Path("nsgvd-local") / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        print("extracted", rel)
```

2. If `nsgvd/stats.py` or `detect.py` were not extracted (Cells A/B never added), create them from Appendix B1/B2.
3. Get guided-diffusion at the pinned commit:

```powershell
git clone https://github.com/openai/guided-diffusion.git nsgvd-local\third_party\guided-diffusion
git -C nsgvd-local\third_party\guided-diffusion checkout 22e0df8183507e13a7813f8d38d51b072ca1e67c
```

4. Check every extracted file compiles (`python -m py_compile <file>`). The notebook's `config.py` may still contain experiment settings (`FRAME_WINDOW = 100_000`); this is harmless because `detect.py` forces the trained value.
