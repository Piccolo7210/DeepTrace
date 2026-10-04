# AIGVDet — hand-off to the local-machine Claude

**Read this whole file before touching anything.** It was written by the Claude instance that
did the Kaggle (GPU) side of this project, for the Claude instance that works on the user's
Windows laptop. You have not seen the earlier conversation; everything you need is here.
When this file and `AIGVDET/PLAN.md` disagree, **this file wins** (it is newer and reflects
measured results). When something is marked **STOP**, do not continue — report to the user.

---

## 0. What you must do, in order

1. **Bootstrap** the local copy: download the code zip and the model export from Google Drive,
   verify checksums (Task 1).
2. **Prove it works here**: run the test suite (expect `12/12`) and the **parity check**
   (`detect_aigvdet.py --parity`, expect `PASSED`) — this is plan gate **G5** (Task 2).
3. **Run real clips**, record seconds per mode — plan gate **G6** (Task 3).
4. Implement **`plot_report`** (Task 4) and the **app integration** (Task 5, plan §10).
5. Write **README.md / STATUS.md** and extend tests (Task 6).
6. **Report back** to the user using the template in section 11.

Do not retrain anything, do not change model constants, do not loosen any tolerance.

---

## 1. The project in one paragraph

Reproduction of *AI-Generated Video Detection via Spatio-Temporal Anomaly Learning*
(Bai, Lin, Cao; arXiv 2403.16638). Two ImageNet-pretrained ResNet-50 binary classifiers: one on
**RGB frames**, one on **RAFT optical-flow colour maps**; per-frame-pair probabilities are
fused `p = α·p_rgb + (1−α)·p_flow` (α = 0.5) and averaged over the clip; the clip is flagged
if `p > τ`. Training and benchmarking ran on Kaggle (2× T4). **Your job is CPU inference on the
user's laptop and integration into their Tk app**, as a second option beside the existing
NSG-VD detector (which must keep working untouched).

---

## 2. Division of labour

| Where | Who | Does |
|---|---|---|
| Kaggle (GPU) | Kaggle-side Claude (another instance) | training, benchmark, calibration, export, speed ablation, any retraining |
| This laptop (CPU) | **you** | pull export, verify, run, time, plot, app integration, docs |

You cannot run GPU work. If a result looks wrong, report it — do not try to "fix" the model.
The user relays messages between the two Claudes by pasting.

---

## 3. What has been done on Kaggle (facts)

### 3.1 Data and training

- Training data: Hugging Face `xmulz/AIGVDet`, revision `c1326befed8ea265e547984e1e25fe83e6b2a7bf`
  (academic use only). Fakes = Moonvalley T2V; reals = YouTube-VOS. 500+500 train videos
  (95 RGB frames and 93 flow maps each → 95,000 / 93,000 training images), 50+50 val videos.
- Recipe (paper + official code): BCE, Adam 1e-4, LR ÷10 after 5 epochs without a ≥0.001
  validation-accuracy gain, stop below 1e-6, batch 64, AMP, random 448 crop, blur/JPEG/flip
  augmentation. One ResNet-50 per branch.
- **RGB branch**: 18 epochs, ~6 GPU-h. Validation accuracy was already 1.0000 at epoch 0, so
  `best.pt` = **epoch 0**; the LR schedule then ran its minimum 18 epochs.
- **Flow branch**: 32 epochs, ~12 GPU-h. Trained on the dataset's **stored** flow maps.
  Best epoch 19: val frame accuracy 0.7857, video accuracy 0.90, video AUC 0.9696.
- The exported weights are the `best.pt` of each branch.

### 3.2 Benchmark on the authors' test videos

Scored with `best.pt`, **K = 16** adjacent-frame pairs per video taken from the first 95 frames,
RAFT-large (torchvision `C_T_V1`, 20 updates), raw (un-conditioned) inputs.
548 fake videos (≤50 per generator set; Sora has 48) vs reals from three pools:
R1 = HF validation reals (50 videos, 1184×636 JPEG frames), R2 = Kinetics-400 (100 videos,
<448 px), R3 = MSR-VTT (100 videos, <448 px). All 798 videos were scored; none failed.

Video-level AUC (RGB-only and flow-only against R1; fused against each real pool):

| Generator set | RGB (R1) | flow (R1) | fused R1 | fused R2 | fused R3 |
|---|---|---|---|---|---|
| T2V Moonvalley (same generator as training) | 1.000 | 0.891 | 1.000 | 0.996 | 0.997 |
| T2V Emu | 1.000 | 0.709 | 1.000 | 0.987 | 0.990 |
| T2V VideoCraft | 1.000 | 0.768 | 0.990 | 0.967 | 0.940 |
| T2V Pika | 0.985 | 0.662 | 0.977 | 0.947 | 0.904 |
| T2V NeverEnds | 0.998 | 0.698 | 0.975 | 0.952 | 0.924 |
| T2V Hotshot | 1.000 | 0.628 | 0.974 | 0.946 | 0.909 |
| T2V Sora | 0.928 | 0.787 | 0.882 | 0.841 | 0.694 |
| T2V VideoPoet | 0.974 | 0.709 | 0.819 | 0.777 | 0.653 |
| I2V Moonvalley | 0.998 | 0.786 | 0.985 | 0.966 | 0.945 |
| I2V NeverEnds | 1.000 | 0.857 | 0.959 | 0.925 | 0.849 |
| I2V Pika | 0.994 | 0.547 | 0.955 | 0.926 | 0.882 |

Other findings: raw inputs beat JPEG-conditioned inputs in all three pools (mean fused AUC
0.956 vs 0.935, 0.930 vs 0.913, 0.881 vs 0.874) → **the app uses raw frames and raw flow
maps, no JPEG round-trip**. The paper's threshold 0.1 flags 26–59 % of real videos here
(unusable); the code's 0.5 flags 0–4 %.

### 3.3 Calibrated threshold (what the app uses)

Chosen from real videos only (calibration reals = R1 + the "calib" half of R2/R3), at 5 %
false-positive rate, following the plan's rule (ship if mean fake recall ≥ 0.60; it is 0.741):

- **τ = 0.429** (95 % bootstrap interval 0.319–0.485), **uncertain band ±0.10**.
  Verdicts: score > 0.529 → "AI-generated"; 0.329–0.529 → "Uncertain"; < 0.329 → "Consistent
  with real". `is_flagged` = plain `score > τ`.
- The 5 % target holds only for the pooled count (R1's 50 reals never false-positive and dilute
  it). On the Drive pools alone, the false-positive rate at τ is **6.8 %** (calibration halves,
  7/103) and **8.2 %** (held-out halves, 8/97); R3 held-out reaches 11.8 %.
- Fake recall at τ per set: Moonvalley T2V 1.00, Emu 1.00, NeverEnds T2V 0.88, Moonvalley I2V
  0.84, Pika I2V 0.82, VideoCraft 0.80, Hotshot 0.76, Pika T2V 0.76, NeverEnds I2V 0.56,
  **VideoPoet 0.40, Sora 0.33**. Detection is uneven across generators — the UI must say so.
- τ belongs to this exact setup: `best` checkpoints, raw inputs, K = 16, α = 0.5. Fewer pairs
  give noisier scores (the detector adds a caveat).

### 3.4 Frozen decisions (do not change)

| Item | Value |
|---|---|
| Crop | 448×448 centre crop, zero-pad if a side is smaller (matches `torchvision.CenterCrop`) |
| Normalisation | ImageNet mean/std, applied after `/255` |
| Optical flow | torchvision `raft_large`, `C_T_V1`, 20 updates; frames replicate-padded to a multiple of 8 (RAFT `InputPadder`); colour map = official `flow_viz` normalised by the frame's max radius; **map keeps the padded size (not cropped back)** |
| Large frames | above 1,500,000 px (`MAX_FLOW_PIXELS`) RAFT runs on a shrunken pair and the field is resized back with vectors rescaled — 1080p needs ~13 GB at native size; this only affects ≥1080p clips |
| Fusion | α = 0.5; clip score = mean over pairs |
| Inputs | raw (no JPEG conditioning) |
| Threshold | read from `aigvdet_meta.json` → `threshold.value`; never hard-code |

### 3.5 Honesty list — known limits (the app and README must not hide these)

1. **The flow branch is weak out-of-domain.** Flow-only AUC is far below RGB-only on most sets
   (e.g. Pika T2V 0.66 vs 0.985). The plan's gate G4 treats that as a sign the flow recipe may
   be wrong; the check (Kaggle "Cell 21 / G1": stored maps vs our maps, same videos, same
   pair count) **has not been run yet**. See 3.6.
2. Sora and VideoPoet are poorly detected (recall 0.33 / 0.40). Sora was scored with flow at
   reduced scale (1080p clips).
3. R1 reals are JPEG frames from the training source (easy); R2/R3 are low-resolution
   (<448 px, zero-padded crops) — resolution is confounded with realness there. No pool stands
   in for modern phone/YouTube footage; **no such real footage was measured**.
4. The benchmark used the first 95 frames; the app spreads pairs over the whole clip (plan
   D14). Clips above ~36 fps and reduced-flow modes are unvalidated.
5. Comparisons with the paper's Table III are indicative only (their reals were GOT-10k).

### 3.6 Pending on the Kaggle side (may change what you download)

- **G1 / flow-recipe check.** If it fails, the flow branch is retrained and the export
  **replaces** the current weights. Until the user tells you otherwise, treat the export as
  **provisional**: write code that reads everything from the bundle, and expect a re-pull.
- Optional: compare `last` vs `best` checkpoints; speed ablation of reduced-scale flow
  (decides whether `standard`/`quick` modes keep their accuracy); K sweep; α sweep.
- The user will tell you when a new export exists. Re-run Task 1 and Task 2 after any re-pull.

---

## 4. Where everything is

Google Drive remote `gdrive:` (rclone), root `gdrive:SPL3/Video-AIGVDet/`:

| Path | Contents |
|---|---|
| `code/aigvdet_code.zip` | the whole local code drop (package, CLI, tests, pull script) |
| `export/aigvdet_rgb.pth`, `export/aigvdet_flow.pth` | plain fp32 state-dicts (~94 MB each) |
| `export/raft_large_C_T_V1.pth` | RAFT-large weights (~21 MB) |
| `export/aigvdet_meta.json` | threshold, calibration, input/flow settings, per-branch epoch and val metrics, `known_limits` |
| `export/parity_samples.npz` | 5 videos' frame pairs + the probabilities Kaggle computed (~22 MB) |
| `export/SHA256SUMS.txt` | `sha256␣␣relative/path` per file |
| `export/results/` | `benchmark_best_k16_raw.csv`, `thresholds_best_k16_raw.json` (documentation) |
| `results/`, `manifests/`, `ckpts/` | Kaggle-side working data — **not needed locally** |

Local target: `AIGVDET\` (project folder) with `AIGVDET\models\` for the export (git-ignored).

---

## 5. Package layout and conventions

After unzipping into `AIGVDET\`:

```
AIGVDET\
  aigvdet\                 one implementation, shared with Kaggle (import as aigvdet.*)
    config.py              constants; the "inference contract" block must not change
    preprocess.py          centre crop/pad, JPEG round-trip (unused locally), normalisation
    models.py              ResNet-50 binary head, load_branch (accepts plain/wrapped/'module.' state-dicts)
    flow_viz.py            vendored RAFT colour wheel (MIT)
    flow.py                load_raft, pad_to_multiple_of_8, flow_scale/clamp_scale/resize_flow, flow_color_map
    video_io.py            probe_video, read_frames, read_selected_frames, pick_pair_indices, resolution_bucket
    scoring.py             PairScorer (RAFT -> maps -> both ResNets), conditioned/scale_limit options
    fuse.py                fuse(), video_score(), verdict()
    verification.py        sha256, write/verify_checksums, parity samples, verify_parity
    detector.py            load_detector(), AIGVDetDetector.analyze()/parity(), MODES, ExportError
    data.py train.py extract.py manifest.py evaluate.py calibrate.py metrics.py export.py remote.py
                           Kaggle-only (kept so there is one source of truth); the detector never imports them
  detect_aigvdet.py        CLI
  tools\pull_export.ps1    Drive export/ -> models\ with SHA-256 verification
  tests\test_aigvdet_module.py
```

Conventions to keep (the user's project style): module-level constants, no dataclass config,
bare names inside a module, package imported as `aigvdet.*`; type hints and docstrings;
`ruff`-clean at 100 columns; PowerShell scripts must be **Windows PowerShell 5.1** compatible
(no `&&`, `||`, ternary). Do **not** modify anything under `Video/` (the NSG-VD detector).

---

## 6. Contracts you can rely on

### 6.1 CLI

```
python detect_aigvdet.py CLIP [--mode benchmark|accurate|standard|quick]
                              [--window whole|first95] [--pairs N] [--json]
                              [--models-dir DIR] [--no-verify] [--threads N]
python detect_aigvdet.py --parity
```
Exit codes: 0 ok / parity passed; 1 error or parity failed; 2 usage.

### 6.2 Modes (provisional except `benchmark`)

| Mode | flow scale | RAFT updates | pairs | status |
|---|---|---|---|---|
| `benchmark` | 1.0 | 20 | 16 | the validated protocol (with `--window first95` it also matches the frame window) |
| `accurate` | 1.0 | 20 | 8 | native flow, fewer pairs → noisier |
| `standard` | 0.75 | 12 | 12 | **accuracy unmeasured** |
| `quick` | 0.5 | 12 | 8 | **accuracy unmeasured** |

The effective flow scale is raised automatically so a shrunken frame never drops below RAFT's
128 px minimum; the scale actually used is in `result["flow_scales"]`.

### 6.3 Result dict of `AIGVDetDetector.analyze(path, mode, window, progress, pairs, ...)`

`p_fake`, `p_spatial`, `p_temporal`, `alpha`, `threshold`, `threshold_source`, `uncertain_band`,
`verdict` ("AI-generated" / "Uncertain" / "Consistent with real"), `verdict_level`
("flagged" / "uncertain" / "clear"), `is_flagged`, `pairs_used`, `pair_indices`,
`frame_probs{spatial,temporal,fused}`, `flow_scales`, `fps`, `resolution [w,h]`,
`frames_in_window`, `duration_seconds`, `mode`, `window`, `settings{pairs,flow_scale,updates}`,
`seconds`, `caveats[]` (plain-language reasons this clip may differ from the benchmark),
`limits[]` (from the bundle's `known_limits`). `progress(done, total)` is called after each pair.
Clips with fewer than 5 decodable frames raise `ValueError`.

### 6.4 Bundle integrity

`load_detector()` verifies every file against `SHA256SUMS.txt` before `torch.load` and raises
`ExportError` (message mentions `pull_export.ps1`) on a missing/truncated/corrupt file. A
truncated model file once silently broke the NSG module — keep verification on by default.

---

## 7. Local environment (from the plan; verify with `pip list`, do not assume)

- Windows 11, Intel i5-1335U (10 cores / 12 threads), ~15.7 GB RAM, Intel Iris Xe (no CUDA) →
  **CPU only**.
- Project on `E:`. Interpreter: `E:\Paper\SPL3\SPL3Full\venv312\Scripts\python.exe`
  (CPython 3.12). **Always call that path** — bare `python` is a different 3.13. The venv's
  console launchers (`pip.exe` etc.) are broken; use `python -m pip`.
- torch 2.13.0+cpu, torchvision 0.28.0+cpu, OpenCV 4.10 with FFmpeg, numpy, scipy, pandas,
  Pillow, matplotlib, tkinter. The detector needs only torch, torchvision, numpy, pandas,
  OpenCV (no scipy/sklearn/huggingface).
- rclone: `D:\rclone-v1.75.1-windows-amd64\rclone-v1.75.1-windows-amd64\rclone.exe`, remote `gdrive:`.
- If PowerShell blocks `.ps1` files: `powershell -ExecutionPolicy Bypass -File <script>`.

---

## 8. Tasks

### Task 1 — Bootstrap and verify

```powershell
$rclone  = "D:\rclone-v1.75.1-windows-amd64\rclone-v1.75.1-windows-amd64\rclone.exe"
$project = "E:\Paper\SPL3\SPL3Full\AIGVDET"          # confirm with the user if different
$python  = "E:\Paper\SPL3\SPL3Full\venv312\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $project | Out-Null
& $rclone copyto gdrive:SPL3/Video-AIGVDet/code/aigvdet_code.zip "$env:TEMP\aigvdet_code.zip"
Expand-Archive -Force "$env:TEMP\aigvdet_code.zip" -DestinationPath $project   # or: tar -xf ... -C $project
& "$project\tools\pull_export.ps1"
```
**Done when:** `pull_export.ps1` prints `ok` for every file and ends with
`Export verified in ...\models`. Existing local files (PLAN.md, PDFs, etc.) must be untouched.
Add `AIGVDET/models/` and `AIGVDET/work/` to `.gitignore`.
**STOP** if any file is `MISSING`/`BAD HASH` after a second attempt, or `rclone` cannot reach
Drive (see section 12 about credentials).

### Task 2 — Prove it works on this machine (gates G5, G8-part)

```powershell
& $python "$project\tests\test_aigvdet_module.py"       # expect: 12/12 tests passed
& $python "$project\detect_aigvdet.py" --parity          # expect: ... PASSED
```
`--parity` re-scores 15 stored frame pairs with the real weights and compares to Kaggle's
probabilities; the largest |Δp| must be ≤ **0.03** (Kaggle itself measured 0.00000). Expect a
tiny CPU-vs-GPU difference (~1e-5–1e-3).
**STOP** (do not proceed to Tasks 3–5) if tests fail or parity fails; section 10 lists what to
check first. **Never raise the tolerance or edit the stored samples.**

### Task 3 — Real clips and timing (gate G6)

1. Pick 3 clips with the user: one 720p+ clip, one ~1184×672 clip, one small (≤320 px) clip.
   The user's I2V test clips are on this PC under `Video\Test_Videos\I2V\…`.
2. For each clip run `--mode accurate` (and `--mode benchmark --window first95` once, to
   compare with the Kaggle protocol). Record: wall-clock seconds, `pairs_used`, resolution,
   `p_fake`, `verdict`, `caveats`, and per-pair `seconds` (total / pairs).
3. The plan measured RAFT at ~17 s per native 1184×640 pair on this laptop (8–17 s depending on
   thermal state): expect `accurate` (8 pairs) to take a few minutes. Target for a future
   `standard` mode is ≤ 90 s — **but `standard`/`quick` are unvalidated; do not make them the
   app default and always show their caveats** until the user says the Kaggle speed ablation
   has confirmed them.
4. Close other heavy programs, note whether the laptop is on mains power, and try
   `--threads 6/10/12` once; report the fastest setting.

### Task 4 — `plot_report(result, out_png)` (not implemented yet)

Put it in `aigvdet/report.py` (matplotlib, **Agg backend forced**), and call it from the CLI
with a `--plot PATH` option. Three panels (plan §9): (a) per-pair `p_spatial` and `p_temporal`
along the clip (x = pair index/time); (b) the RGB crop and flow crop the networks saw for the
most and least suspicious pair; (c) fused clip score against τ with the ±0.10 uncertain band
shaded. Panel (b) needs the crops: add an opt-in `keep_crops=True` argument to
`AIGVDetDetector.analyze` (store only the two extreme pairs' 448×448 crops, as uint8) — do not
store crops by default. Add a test that renders a figure from a synthetic result.

### Task 5 — App integration (plan §10; touch only what is listed)

1. `app/paths.py`: add `AIGVDET_DIR`.
2. **New** `app/ui/video_menu_view.py`: chooser opened by the existing Video Forensics button —
   two cards, "NSG-based (diffusion physics)" and "AIGVDet-based (RGB + optical flow)", each
   showing model readiness and a typical runtime; Back to Home.
3. `app/ui/video_view.py`: add a *detector profile* (engine attribute, header title,
   instruction text, limitation text, measurement rows, figure caption, progress style). The
   default profile must reproduce today's NSG behaviour **exactly**. The AIGVDet profile adds:
   per-branch probabilities (RGB / flow / fused), a "pairs analysed" line, a **determinate
   progress bar** (driven by `progress(done, total)`), the `caveats` list, the `limits`, and a
   banner when the clip's short side is < 448 px (training crop size).
4. **New** `app/aigvdet_engine.py`: copy the shape of `video_engine.py` — `load_async`,
   `analyze_async` with an identity `request_id`, one video at a time, `new_run_dir` — plus an
   `analysis_progress` event `(done, total)`. Run inference in a worker thread; never block the
   Tk main loop; cancel/ignore stale `request_id`s.
5. `app/ui/app_window.py`: construct the engine, call `load_async()`, add it to the
   `_poll_events` tuple, add `show_video_menu()` / `show_aigvdet()`.
6. `app/ui/home_view.py`: the Video Forensics button now calls `show_video_menu`;
   `_status_text` adds "AIGVDet models ready" (use `ExportError`/missing models to show "run
   pull_export.ps1" instead of crashing).
7. Results go to a run dir under `results/app/` (same convention as NSG).
8. The app must show the verdict **with its caveats**; never present `p_fake` as a calibrated
   probability — it is a score compared with a threshold.
**Done when (gates G7, G8):** Video Forensics → chooser → AIGVDet → pick a clip → progress →
verdict + figure + run dir; the NSG path still works end to end; both
`AIGVDET\tests\test_aigvdet_module.py` and the existing `Video\tests\test_video_module.py` pass.

### Task 6 — Documentation and tests

- `AIGVDET/README.md` (how to pull, run, limits) and `AIGVDET/STATUS.md` (what is done, gate
  results G5–G8, timings, open items), modelled on `Video/README.md` and `Video/STATUS.md`.
  README must include the honesty list (3.5) and the dataset licence note (section 12).
- Extend `tests/test_aigvdet_module.py` for anything you add (report, app engine with a fake
  detector). Keep tests runnable without weights.

---

## 9. Rules

**Do**
- Read thresholds, α, flow settings and input policy from `aigvdet_meta.json` (the detector
  already does).
- Keep checksum verification on; keep CPU-only; keep results reproducible.
- Show caveats and limits to the end user; say "uncertain" when the verdict is uncertain.
- Keep changes small and listed; run the tests after each change.

**Do not**
- Change `CROP_SIZE`, mean/std, `FUSION_ALPHA`, RAFT settings, `MAX_FLOW_PIXELS`, the verdict
  logic, or the JPEG policy. Do not hard-code 0.429 anywhere.
- Retrain, fine-tune, or "improve" accuracy locally; do not tune α/τ on test videos.
- Loosen the parity tolerance (0.03) or edit `parity_samples.npz`.
- Edit anything under `Video/` (NSG-VD) or commit `models/`, `*.pth`, `rclone.conf`.
- Make `standard`/`quick` the default or describe their accuracy as known.
- Claim detection of real-world phone/YouTube footage is validated — it is not.

---

## 10. STOP conditions and first-aid

| Symptom | Check first | Then |
|---|---|---|
| `pull_export.ps1`: BAD HASH / MISSING | re-run once (interrupted download) | STOP, report file names |
| Tests fail on import | wrong interpreter (use `venv312`), missing package | report traceback |
| `ExportError` at load | run `pull_export.ps1` again; never use `--no-verify` to "make it work" | report |
| `--parity` FAILED | confirm checksums OK; torch/torchvision versions vs `aigvdet_meta.json`; try `--threads 1`; run once with MKLDNN disabled (`torch.backends.mkldnn.enabled = False`) to see whether oneDNN is the cause | STOP, report the per-pair table and versions |
| `error: ... frames are too small for RAFT` | clip's short side < 128 px — expected, report as unsupported | — |
| Very slow (> 20 s/pair) | thermal throttling, power plan, other processes, thread count | report timings |
| Verdict looks wrong on a clip | it may be a known weak spot (3.5); read the caveats | report clip type + result; do not change code |

---

## 11. Report-back template (user pastes this to the Kaggle-side Claude)

```
ENV     torch/torchvision versions, opencv version, python path used
TASK 1  pull_export: all files ok? (yes/no) – export meta created_utc and code_hash
TASK 2  tests: N/N passed | parity: PASSED/FAILED, largest |dp| = ...  (paste the table if FAILED)
TASK 3  per clip: resolution, fps, mode, pairs, seconds, seconds/pair, p_fake, verdict, caveats
        fastest --threads setting
TASK 4/5/6  what was implemented, files changed, gate G7/G8 status, open problems
```

---

## 12. Security and licence notes

- **Credentials:** the rclone config (OAuth client secret + refresh token) was accidentally
  exposed in a chat earlier. The user should rotate it: revoke the old token, reset the OAuth
  client secret in Google Cloud Console, then run `rclone config reconnect gdrive:` locally
  and update the Kaggle `RCLONE_CONF` secret. If rclone suddenly fails with auth errors, this is
  the likely reason — tell the user. Never print, log or commit `rclone.conf`.
- **Licence:** the training dataset is academic-use-only; models trained on it should be
  described as research artefacts in README/UI. The vendored `flow_viz.py` is MIT (T. Runia);
  RAFT weights come from torchvision.

---

## Appendix — gates referenced

| Gate | Meaning | Where | Status |
|---|---|---|---|
| G1 | flow branch accepts maps made by our RAFT recipe | Kaggle | **pending** (read-out from the benchmark was marginal: flow AUC 0.891 vs stored-map 0.970) |
| G4 | results sanity (Moonvalley AUC ≥ 0.95 ✔; T2V mean AUC ≥ 0.85 ✔; flow not far below RGB ✘) | Kaggle | partly failed → G1 |
| G5 | Kaggle ↔ local parity, \|Δp\| ≤ 0.03 | **local (you)** | to do |
| G6 | per-mode CPU timing recorded | **local (you)** | to do |
| G7 | app smoke test | **local (you)** | to do |
| G8 | both test suites pass | **local (you)** | to do |
