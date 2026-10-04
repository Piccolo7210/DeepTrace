# AIGVDet module — status

**As of 4 October 2026.** The local side is built, verified on this laptop and wired into the
app. What is still open is on the Kaggle side (flow-recipe check, possible re-export) and in
validation that needs data we do not have. How it works: `README.md`. Plan: `PLAN.md`.
Hand-off tasks followed: `AIGVDET_LOCAL_HANDOFF.md`.

## 1. Gates

| Gate | Result |
|---|---|
| G5 Kaggle ↔ local parity | **PASSED** — 15 stored frame pairs, largest \|Δp\| = **0.00983** (tolerance 0.03); RGB branch differences 0.0 |
| G6 CPU timing | recorded, section 3 |
| G7 app smoke test | engine run end to end through the app's own import path on a real clip (progress 1/8…8/8 → verdict, figure, `result.json`); real `AppWindow` navigated Home → chooser → AIGVDet → NSG screen; every state of both screens rendered (instructions, selected, running with progress, results, error). **Not done: a human click-through of the file dialog** — `askopenfilename` cannot be driven from a script |
| G8 tests | `AIGVDET/tests/test_aigvdet_module.py` **16/16**; `Video/tests/test_video_module.py` **23 passed, 0 failed, 0 skipped** |

## 2. What was added

| Where | What |
|---|---|
| `AIGVDET/aigvdet/report.py` | `plot_report(result, out_png)` — Agg backend, three panels |
| `AIGVDET/aigvdet/detector.py`, `scoring.py` | opt-in `keep_crops=True` (crops of only the most and least suspicious pair); `result["calibration"]` (the bundle's test recall numbers, for UI text). No logic or constant changed |
| `AIGVDET/detect_aigvdet.py` | `--plot PATH` |
| `app/aigvdet_engine.py` | **new** — worker thread, `analysis_progress` events, one video at a time, stale `request_id`s ignored by the view, `result.json` + `analysis.png` in a `results/app/` run dir |
| `app/ui/video_menu_view.py` | **new** — the Video Forensics chooser (NSG-based / AIGVDet-based cards with model readiness) |
| `app/ui/video_view.py` | profile-driven: `NSG_PROFILE` (default) and `AIGVDET_PROFILE`; AIGVDet adds the mode choice, determinate progress bar, per-branch scores, caveats, limits, and the small-clip banner |
| `app/ui/app_window.py`, `home_view.py`, `paths.py` | engine construction and polling, `show_video_menu()` / `show_aigvdet()`, the Home button opens the chooser, status line mentions AIGVDet, `AIGVDET_DIR` |
| `.gitignore` | `AIGVDET/models/`, `AIGVDET/work/` |
| `tests/test_aigvdet_module.py` | + report rendering, `keep_crops`, engine with a fake detector (progress, errors, busy rejection, failed load), view smoke test (skipped without a display) |

Notes on decisions made while integrating:

- **`import torch` first in `app/aigvdet_engine.py`.** Importing pandas before torch breaks torch
  here (`WinError 1114`); `aigvdet.detector` imports pandas, so without this the engine failed to
  load in the app while the CLI and tests (which import torch first) worked. The shared package
  was left untouched so a re-pull cannot undo the fix.
- Only the *imports* run under the shared import lock; the weights load outside it, because the
  other engines hold that lock for their whole model load (AIGVDet is ready as soon as the image
  engine releases the lock, ~20 s after start).
- The default mode is `accurate`; `standard`/`quick` are selectable but labelled "accuracy NOT
  measured", and the detector adds its reduced-scale caveat to their results.
- `app/ui/video_view.py` on disk had no limitations banner or caveats section for the NSG screen
  (it differed from the version read earlier in the session); that was left as found, so the NSG
  profile shows none. Header and back button changed for both screens: "← Back" returns to the
  chooser.

## 3. Timings (gate G6)

Intel i5-1335U, torch 2.13.0+cpu, 10 threads (torch default), `--mode accurate` = 8 pairs spread
over the clip. Mains power and background load were not controlled or recorded.

| Clip | Resolution · fps | Mode | Pairs | Seconds | s / pair | Score | Verdict |
|---|---|---|---|---|---|---|---|
| Sora (T2V sample) | 1280×720 · 30 | accurate | 8 | 108.6 | 13.6 | 0.125 | Consistent with real (Sora is a known weak spot, recall 0.33) |
| Moonvalley T2V | 1184×672 · 16 | accurate | 8 | 87.7 | 11.0 | 0.676 | AI-generated |
| Moonvalley T2V | 1184×672 · 16 | benchmark, `first95` | 16 | 182.1 | 11.4 | 0.679 | AI-generated |
| Moonvalley I2V | 626×626 · 16 | accurate | 8 | 43.3 | 5.4 | 0.518 | Uncertain |
| Pika I2V (via the app engine) | 960×640 · 24 | accurate | 8 | 68.0 | 8.5 | 0.797 | AI-generated |
| MSR-VTT (real) | 320×240 · 25 | accurate | 8 | 9.9 | 1.2 | 0.143 | Consistent with real |

The 8-pair `accurate` score (0.676) agrees with the 16-pair benchmark protocol (0.679) on the one
clip run both ways — one clip, not a validation.

Threads (Moonvalley I2V, accurate): 6 → 56.9 s, 10 → 45.8 s, 12 → 45.1 s. **10 and 12 are
equivalent; the torch default (10) is kept.**

Model load: ~1–3 s on its own (checksums + weights).

## 4. Open items

1. **Export is provisional.** The Kaggle-side flow-recipe check (G1) is pending; if it fails the
   flow branch is retrained and the export replaces the current weights. Then: `pull_export.ps1`,
   tests, `--parity`, and the timings above should be spot-checked again.
2. `standard` / `quick` accuracy is unmeasured (Kaggle speed ablation). Until then they stay
   non-default and labelled.
3. No real phone/YouTube footage has been scored; the false-positive rate on such clips is unknown.
4. **Credentials:** the rclone config was exposed in an earlier chat. Rotate it (revoke the token,
   reset the OAuth client secret, `rclone config reconnect gdrive:`, update the Kaggle
   `RCLONE_CONF` secret). An auth failure in `pull_export.ps1` is the likely symptom.
5. Optional: a human click-through of Home → Video Forensics → AIGVDet-based → Choose Video…

## 5. Report back to the Kaggle-side Claude

```
ENV     torch 2.13.0+cpu / torchvision 0.28.0+cpu, opencv 4.10.0, python E:\Paper\SPL3\SPL3Full\venv312\Scripts\python.exe (3.12.6)
TASK 1  pull_export: all files ok (yes) – export created_utc 2026-10-04T00:17:17+00:00, code_hash 50b724d8bab52692
TASK 2  tests 16/16 passed (13/13 before the additions) | parity PASSED, largest |dp| = 0.00983
TASK 3  see section 3; fastest threads: 10-12 (equal)
TASK 4/5/6  report.py + --plot, app engine/chooser/profiled view, README/STATUS, extra tests;
        G7 automated (human file-dialog click-through not done), G8 16/16 + 23 passed
        problem found: pandas-before-torch breaks torch on this PC (worked around in app/aigvdet_engine.py)
```
