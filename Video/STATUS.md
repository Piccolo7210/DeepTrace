# Video Forensics Module — Status

**As of 2 October 2026.** Where the NSG video detector stands, what was built, and what
is still open. For *how* the method works see `NSGVD_KAGGLE_HANDOFF.md` (the authoritative
spec, written from the Kaggle run) and `README.md` (usage). `PLAN.md` is the original
project plan — see the note in §6 before trusting its local-inference section.

---

## 1. Short version

The module is **functionally complete and wired into the app**. Kaggle training is
finished, the trained detector and the 2.1 GB score network are on this machine, local
inference runs on the CPU, and the app's **Video Forensics** button is live. What remains
is validation-and-polish work, not construction: a wild-video false-positive check, an
end-to-end run through the UI by hand, and the first commit (the whole `Video/`, `app/`
and `Speech/` trees are still untracked).

---

## 2. Done — Kaggle side (summarised from the handoff)

| | |
|---|---|
| Dataset | AIGVDet, split 8 000 dev/reference + 1 000 validation + 2 000 unseen-generator test |
| Features | 8 frames/clip, NSG from a frozen guided-diffusion UNet, 62 per-clip statistics |
| Detector | `HistGradientBoostingClassifier`, 578 trees, baseline 0.15415068 |
| Threshold | τ = **0.6915**, calibrated for 5% false positives on validation reals (held at 4.8% on test reals) |
| Performance | validation AUROC 0.969 · test AUROC 0.838 · 40.9% of generated clips caught at τ |
| Not used | the paper's deep-kernel MMD head — reproduced faithfully, did not learn on this data (val AUROC 0.554) |

The detector is therefore **NSG features + a supervised gradient-boosting head**. It must
not be described as the paper's MMD method, nor as pure physics: in the training data,
resolution and frame rate alone separated the two classes, so some of the accuracy is
plausibly a codec/resolution shortcut. That caveat is carried in every result
(`config.py: CAVEATS`); the app no longer displays it.

---

## 3. Done — local inference

| Path | Role |
|---|---|
| `nsgvd_local/` | the Kaggle bundle, **unmodified** — frame sampling, score, NSG, the 62 statistics, vendored `guided-diffusion` @ `22e0df8` |
| `nsgvd_local/models/` | weights, git-ignored: UNet 2.1 GB · `nsg_stats_gbm.joblib` · `parity_samples.npz` |
| `config.py` | module constants; the "inference contract" block is fixed, not tunable |
| `predictor.py` | sklearn-free evaluation of the 578 trees |
| `models/nsg_gbm_trees.npz` | those trees re-encoded for numpy (0.4 MB, regenerable) |
| `tools/export_tree_dump.py` | produces and verifies that file from the `.joblib` |
| `detect_nsgvd.py` | app-facing wrapper: `load_detector()`, `detect_video()`, `plot_report()`, CLI (`--parity`, `--plot`, `--json`) |
| `tests/test_video_module.py` | 4 groups: classifier, stored-feature regression, frame sampling, parity + end-to-end |

Two problems were solved along the way and are worth not re-discovering:

- **The scikit-learn clash.** The pickled classifier only unpickles under scikit-learn
  1.6.1 (1.9.0 fails with `No module named '_loss'`), but the image and speech modules
  were built against 1.9.0. Rather than pin the whole project to one module's needs, the
  trees were exported once under 1.6.1 and are now evaluated with numpy. The re-encoding
  was checked against scikit-learn's own `predict_proba` on 2 207 rows (random,
  all-zero, all-NaN, and the 5 parity clips): **max |Δp| = 1.1e-16**. The project runs
  1.9.0 again and this module needs no sklearn at all.
- **Truncated weights read as a corrupt model.** A half-finished download surfaced as
  `PytorchStreamReader failed … checkpoint file is corrupted`. `config.py` now records the
  exact byte count (2 211 383 297) and `unet_status()` checks it before `torch.load`, so
  an unfinished transfer says so plainly and the tests skip instead of failing.

---

## 4. Done — app integration

- `app/video_engine.py` — `VideoForensicsEngine`, mirroring `speech_engine.py`: loads the
  detector once on a background thread at startup (~2.2 GB RAM), runs each analysis on a
  worker thread, posts `("analysis_done"|"analysis_failed"|"models_ready"|"models_failed")`
  to the shared queue with an identity-guarded `request_id`. **One video at a time** — a
  second concurrent UNet pass would thrash a 16 GB machine, so it is rejected with a clear
  message rather than queued.
- `app/ui/video_view.py` — the same five-state view as speech (instructions → selected →
  running → results → error): metadata and thumbnail probe on selection, a warning when
  the clip has fewer than 8 decodable frames (Analyze disabled) or more than 95 (states
  what portion was judged), verdict banner at three levels (**AI-generated** /
  **Uncertain**, within 0.10 of τ / **Consistent with real**), a measurements table, the
  three-panel report figure.
- `app/paths.py`, `app/ui/app_window.py`, `app/ui/home_view.py` — `VIDEO_DIR`, engine
  construction and startup load, `show_video_forensics()`, the poll loop extended to all
  three engines, and the Home button changed from "coming soon" to live.
- The other two modules were left alone. The `isolated_import` shim was re-verified
  three ways (`src/config.py` → `(512, 'cpu')`, `Speech/config.py` → `(True, 'cpu')`,
  `Video/config.py` → `(95, 224)`) with `sys.path` clean afterwards and no stale bare
  `config`/`detect` left in `sys.modules`. `detect_nsgvd.py` adds `nsgvd_local/` to
  `sys.path` only inside a context manager, because that directory has its own top-level
  `detect.py` which would otherwise hijack `import detect`.

---

## 5. Verified

| Check | Result |
|---|---|
| Fast tests (classifier, stored features, frame sampling) | **16 passed, 0 failed** |
| Tree dump vs scikit-learn `predict_proba`, 2 207 rows | max \|Δp\| = 1.1e-16 |
| Stored-feature probabilities vs recorded values | regression-pinned in the test file |
| Canonical sampler | `sample_indices(95) == [0, 11, 22, 33, 44, 55, 66, 77]` |
| Short clip rejected | `VideoDecodeError: only 4 decodable frames (< 8)` |
| Three-way module isolation | clean |
| Weights, secrets and the dataset PDF out of git | confirmed with `git check-ignore -v` |
| UNet download | complete, 2 211 383 297 bytes, recorded SHA-256 `a37c32fffd31…` |
| Full suite incl. parity + end-to-end | **23 passed, 0 failed, 0 skipped** |
| Parity vs Kaggle's stored fp16 GPU features (5 clips) | largest diff **0.0016** (tolerance 0.05) |
| End-to-end timing on this CPU | synthetic clip: **109 s**; a real 2.9 s/36-frame clip: **45 s** |
| Real-clip check through the actual app code path | `detect_nsgvd.py` == what `video_engine.py` calls (`load_detector`/`analyze`/`plot_report`); ran a real Moonvalley I2V clip end to end: decode → 8-frame sample → UNet → 62 stats → classifier → verdict, figure written |

The five clips in `parity_samples.npz` are all Hotshot, and all five score *below* τ
(0.451, 0.103, 0.098, 0.043, 0.028). That is consistent with the handoff's 14.2% Hotshot
recall and is not a bug — but it does mean the parity file is useless as a demo: it will
never show a positive detection. The one real clip tried above (Moonvalley I2V) also
scored below τ (0.2367) — also not a bug (that generator's measured recall is 68.6%, not
100%), but it means picking a demo clip that actually gets flagged still needs trying a
few from `Video/Test_Videos/` (now on disk, git-ignored) rather than assuming the first
one works.

---

## 6. What is left

The integration itself — Kaggle output wired into this app — is done and verified (§5).
What remains is validation/polish that the module owner, not the integration, should
drive:

1. **Click through the UI by hand** — `venv312\Scripts\python -m app` → Home → Video
   Forensics → pick a clip → progress → verdict banner, figure, and a run directory under
   `results/app/<ts>_<stem>/`. The engine-level path (`load_detector`/`analyze`/
   `plot_report`) is confirmed working from the CLI; only the tkinter click-through itself
   is unconfirmed.
2. **Pick a demo clip that actually gets flagged.** `Video/Test_Videos/` (user-provided,
   git-ignored, 3 000 I2V clips across moonvalley/neverends/pika) is on disk now. The one
   tried so far (a Moonvalley I2V clip) scored under τ — not a bug, just a miss, see §5.
3. **Wild-video false-positive check** (handoff §9) — 10–20 ordinary high-resolution phone
   videos, none of them AI, to test the resolution-shortcut worry in §2. Needs clips the
   module owner supplies; not something to source independently.
4. **Reconcile `PLAN.md` §7** (local side) — written for the abandoned MMD bundle,
   superseded by the handoff. §§1–6 on the Kaggle side are still accurate.
5. **Commit.** `Video/`, `app/` and `Speech/` are all still untracked, alongside modified
   `src/` files.
6. *Optional, proposed but not built:* a filmstrip of the 8 scored frames in
   `plot_report`.

---

## 7. Do not break these

From handoff §10, and the reason each one exists:

- **Do not change any constant in the inference contract** (§7 of the handoff / the marked
  block in `config.py`): frame count, window, resolution, watermark crop, the score
  definition, the denominator, the statistics. The features would stop matching the trained
  model, and the failure mode is silently wrong probabilities, not an exception.
- **Do not retrain or fine-tune locally.** This machine runs inference only.
- **Do not upgrade scikit-learn or drop numpy below 2.0** without re-exporting the model on
  Kaggle.
- **Do not use** `export/nsgvd_local_res224.pt` (the failed MMD model) or the `_full`
  feature folders.
- **Never commit secrets** — `rclone.conf`, OAuth tokens, `client_secrets.json`,
  `credentials.json`. If one is ever exposed, revoke and recreate it. Keep the 2.1 GB
  weights out of git.
- **Describe the detector accurately:** NSG features plus a gradient-boosting head, with
  the caveats in §2 stated.

---

## 8. Picking it up again

```powershell
# acceptance checks (needs the weights)
venv312\Scripts\python Video\tests\test_video_module.py

# one video, with the report figure
venv312\Scripts\python Video\detect_nsgvd.py my_clip.mp4 --plot

# just the Kaggle-parity check
venv312\Scripts\python Video\detect_nsgvd.py --parity

# the app
venv312\Scripts\python -m app
```

Re-fetching the weights, regenerating the tree dump, and the full performance table are
all in `README.md`.
