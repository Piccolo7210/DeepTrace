# AIGVDet — AI-generated video detection (local, CPU)

Reproduction of *AI-Generated Video Detection via Spatio-Temporal Anomaly Learning*
(Bai, Lin, Cao; arXiv 2403.16638). Two ImageNet-pretrained ResNet-50 classifiers — one on
**RGB frames**, one on **RAFT optical-flow colour maps** — are fused per frame pair
(`p = 0.5·p_rgb + 0.5·p_flow`) and averaged over the clip; the clip is flagged when the
score exceeds a calibrated threshold. Training and benchmarking ran on Kaggle (2× T4); this
folder runs **inference on the CPU** and plugs into the desktop app as the second option of
**Video Forensics** (the first is the NSG-based detector in `Video/`, untouched).

Status, gate results and timings: [`STATUS.md`](STATUS.md). Design: [`PLAN.md`](PLAN.md).
Hand-off from the Kaggle side (wins over the plan where they differ):
[`AIGVDET_LOCAL_HANDOFF.md`](AIGVDET_LOCAL_HANDOFF.md).

## In the app

```powershell
venv312\Scripts\python -m app
```

Home → **Video Forensics** → choose **NSG-based** or **AIGVDet-based** → pick a clip →
pick a mode (default *Accurate*) → progress bar (one step per frame pair) → verdict, scores,
figure and the caveats. Each analysis writes `results/app/<time>_<clip>/` with `analysis.png`
and `result.json`. If the model files are missing the chooser says to run `pull_export.ps1`.

## Command line

```powershell
venv312\Scripts\python AIGVDET\detect_aigvdet.py clip.mp4                    # accurate mode
venv312\Scripts\python AIGVDET\detect_aigvdet.py clip.mp4 --plot report.png  # + figure
venv312\Scripts\python AIGVDET\detect_aigvdet.py clip.mp4 --mode benchmark --window first95 --json
venv312\Scripts\python AIGVDET\detect_aigvdet.py --parity                    # Kaggle <-> local gate
```

Always use `venv312\Scripts\python` (a bare `python` on this PC is a different 3.13) and
`python -m pip` (the venv's `pip.exe` launchers point at a deleted folder).

## Getting the model files

Not in git. With the rclone remote `gdrive:` configured:

```powershell
$rclone = "D:\rclone-v1.75.1-windows-amd64\rclone-v1.75.1-windows-amd64\rclone.exe"
& $rclone copyto gdrive:SPL3/Video-AIGVDet/code/aigvdet_code.zip "$env:TEMP\aigvdet_code.zip"
Expand-Archive -Force "$env:TEMP\aigvdet_code.zip" -DestinationPath AIGVDET   # code (only after a new drop)
powershell -ExecutionPolicy Bypass -File AIGVDET\tools\pull_export.ps1         # models/, SHA-256 verified
```

`models/` holds `aigvdet_rgb.pth`, `aigvdet_flow.pth`, `raft_large_C_T_V1.pth`,
`aigvdet_meta.json` (threshold, calibration, limits) and `parity_samples.npz`. The detector
verifies every file against `SHA256SUMS.txt` before loading and refuses a truncated one.
**The export is provisional**: the Kaggle side may retrain the flow branch; re-run
`pull_export.ps1`, then the tests and `--parity`, after any new export.

## Modes

| Mode | Optical flow | Pairs | Notes |
|---|---|---|---|
| `accurate` (app default) | native, 20 updates | 8 | about 1–2 minutes per clip on this laptop |
| `benchmark` | native, 20 updates | 16 | the validated protocol (with `--window first95` it also matches the frame window); about 3 minutes |
| `standard` | 0.75×, 12 updates | 12 | **accuracy not measured** |
| `quick` | 0.5×, 12 updates | 8 | **accuracy not measured** |

The threshold (0.429, uncertain band ±0.10) was calibrated for 16 pairs; fewer pairs give a
noisier score and the result says so. Measured timings: `STATUS.md`.

## What the verdict means — and does not

The score is **not a probability**; it is compared with a threshold calibrated for about 5 %
false positives on a pool of real videos. Verdicts: *AI-generated* (score above τ + 0.10),
*Uncertain* (within ±0.10 of τ), *Consistent with real*.

Known limits (also shown in the app next to every result):

1. **The flow branch is weak out of domain.** Flow-only AUC is far below RGB-only on most
   test sets (e.g. Pika T2V 0.66 vs 0.985), so most of the detection comes from the RGB branch.
   Whether the flow recipe matches the stored training maps (plan gate G1) is still being
   checked on the Kaggle side.
2. **Detection is uneven across generators.** At the threshold the recall was 1.00 for
   Moonvalley T2V and Emu but **0.40 for VideoPoet and 0.33 for Sora**; mean 0.74.
3. **No real phone or YouTube footage was measured.** Real-video false positives come from
   in-domain YouTube-VOS frames and low-resolution Kinetics/MSR-VTT clips (<448 px,
   zero-padded; resolution is confounded with realness there). On the Drive pools alone the
   false-positive rate at τ was 6.8 % (calibration half) and 8.2 % (held-out half); one pool
   reached 11.8 %.
4. The benchmark used the first 95 frames; the app spreads pairs over the whole clip. Clips
   above ~36 fps and the reduced-flow modes are unvalidated.
5. Comparisons with the paper's Table III are indicative only (its reals were GOT-10k).
6. One training generator (Moonvalley), 2023–24 generators in the test sets: newer
   generators may not be detected.

## Layout

```
aigvdet/             one implementation shared with Kaggle (import as aigvdet.*)
  config.py          constants; the "inference contract" block must not change
  detector.py        load_detector(), AIGVDetDetector.analyze()/parity(), MODES, ExportError
  scoring.py flow.py flow_viz.py preprocess.py video_io.py fuse.py verification.py
  report.py          plot_report(): per-pair scores, the crops the networks saw, score vs threshold
  data.py train.py extract.py evaluate.py calibrate.py ...   Kaggle-only; the detector never imports them
detect_aigvdet.py    CLI
tools/pull_export.ps1
tests/test_aigvdet_module.py   16 tests, no weights needed
models/ work/        git-ignored
```

App side: `app/aigvdet_engine.py` (worker thread, progress events), `app/ui/video_menu_view.py`
(the chooser), `app/ui/video_view.py` (one screen class, NSG and AIGVDet profiles).

## Tests

```powershell
venv312\Scripts\python AIGVDET\tests\test_aigvdet_module.py     # expect 16/16
venv312\Scripts\python Video\tests\test_video_module.py          # NSG module, expect 23 passed
```

## Environment gotcha

On this machine **importing pandas before torch breaks torch** (`WinError 1114`, `c10.dll`
initialisation) — reproducible in plain Python. `aigvdet.detector` imports pandas, so the app
engine imports torch first at app start (`app/aigvdet_engine.py`). Keep that import order in
any new entry point.

## Licence and citation

The training data (Hugging Face `xmulz/AIGVDet`) is released **for academic research only; no
commercial use**. The weights are research artefacts and must be described that way.
`aigvdet/flow_viz.py` is MIT (T. Runia); RAFT weights come from torchvision (BSD-3).

> J. Bai, M. Lin, G. Cao, *AI-Generated Video Detection via Spatio-Temporal Anomaly Learning*,
> arXiv:2403.16638 (PRCV 2024).
