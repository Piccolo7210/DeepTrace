# Why a real phone video scored close to the threshold — and what can be done

**Trigger:** a genuine (non-AI) phone video — portrait, 576×1024, 30 fps — scored
`p_fake = 0.676` against `τ = 0.6915`, landing in the "Uncertain" band (within 0.10 of
the threshold). Not a crash, not a confident "AI-generated" verdict — but close enough
to the line that it's worth understanding, not just shrugging off.

---

## 1. What the evidence says

| Clip | Resolution / fps | Profile vs. training "real" clips | p_fake |
|---|---|---|---|
| `Speech/test_clips/bbac3n_real.mp4` (genuine, low-res talking-head) | 360×288 @ 25 fps | **matches** (training reals: ≤568×320 @ ~30 fps) | **0.177** — confidently real |
| The reported phone video | 576×1024 @ 30 fps | **doesn't match** — long side (1024) almost 2× the training ceiling | **0.676** — uncertain |

Two points, not a study — but they move in exactly the direction the module's own
documented caveat predicts (`Video/config.py: CAVEATS`, `Video/README.md`: *"In the
training data, real and generated clips also differed in resolution and compression, so
part of the accuracy may come from those cues rather than from the physics of the
video."*). This looks like that known weakness showing up on a real input, not a new bug.

### Why resizing every clip to 224×224 doesn't erase this

Every clip — real or fake, any native resolution — is centre-cropped and downscaled to
224×224 before scoring (`Video/config.py: FEATURE_SIZE`). So the model never literally
reads "576×1024." Two things still leak through:

- **Resampling ratio.** A 360×288 clip downscaled to 224 and a 576×1024 clip's centre
  square (576×576) downscaled to 224 go through very different degrees of downscaling.
  That leaves different blur / compression-block signatures in the final 224×224 image —
  exactly the kind of fine-grained statistical regularity a diffusion score network is
  sensitive to.
- **Frame rate changes the physics term itself.** The detector samples frames at fixed
  *indices* (stride through the first 95 frames), not fixed *time offsets*. A 30 fps clip
  and an 8 fps clip cover very different amounts of real-world time between those sampled
  frames, which directly changes the motion magnitude the NSG temporal term measures.
  This one isn't even a "shortcut" exactly — fps genuinely changes the feature, by
  construction of the method.

So this isn't the classifier peeking at metadata it shouldn't. It's that resolution and
frame rate leave real statistical fingerprints that survive the resize, on top of
whatever genuine diffusion-physics signal is also there — and the trained classifier has
no way to tell the two apart.

---

## 2. Options

| # | Option | Fixes the root cause? | Effort | Where |
|---|---|---|---|---|
| A | Do nothing | No | — | — |
| B | Local: add a specific, measured-from-this-clip caveat when a clip's resolution is far outside the training-real profile | No — doesn't touch the score | Small | Here, now |
| C | Kaggle: recalibrate τ using a validation set that includes modern high-res/portrait real footage | Partial — fixes the threshold, not the underlying shortcut | Medium | Kaggle, needs you |
| D | Kaggle: retrain with a resolution/fps-diverse real-video pool, so the classifier can't lean on it | **Yes** — removes the shortcut itself | Large | Kaggle, needs you |
| E | Locally lower τ / widen the uncertain band so phone video reads more comfortably "real" | No — actively bad | — | **Don't** |

### A. Do nothing
Honest option. The result is already labelled "Uncertain," not "AI-generated" — nothing
is factually wrong today. The downside is purely that the generic caveat paragraph
doesn't tell a user *why* this particular clip is borderline, only that the model has
caveats in general.

### B. Sharper local caveat (what I was about to build before you stopped me)
Add a threshold check in `detect_nsgvd.py: analyze()`: if a clip's longer side exceeds
~568px (the documented ceiling among training "real" clips), prepend a specific caveat
naming the measured resolution/fps and saying plainly that this profile is outside what
the detector learned "real" looks like. No model file, threshold, or score changes —
purely a more honest, input-specific explanation shown alongside the existing verdict.
Safe, small, reversible, doesn't touch anything under "do not change" in the handoff.

### C. Recalibrate τ on Kaggle with realistic real-video resolutions
τ = 0.6915 was calibrated for 5% false positives using AIGVDet's own real clips (all
≤568×320). If the validation real-video pool included modern phone-resolution footage,
the calibrated threshold would reflect the actual deployment distribution instead of a
curated one. This needs re-running the calibration step on Kaggle with new real-video
data (your own phone clips, or a public phone-video dataset) — I can't do this from here
(no Kaggle access, and this machine is inference-only per the handoff), but I can help
draft the Kaggle notebook/script for it if you want to run it yourself.

### D. Retrain with resolution/fps-diverse real data
The actual fix: if "real" in training spans the same resolution/fps range as "fake," the
classifier physically cannot use that range as a shortcut anymore, and whatever AUROC
remains reflects the intended physics signal. This is the only option that closes the
gap rather than working around it. It's a bigger lift — new real-video sourcing, a
Kaggle re-run of feature extraction + training (stages K0–K5 in `PLAN.md`) — but it's the
right long-term answer if this module needs to handle arbitrary user-supplied video
rather than only AIGVDet-shaped clips.

### E. Why not just move the threshold locally
Tempting, but wrong: τ is a calibrated statistic (5% FPR on a specific validation set),
not a free dial. Nudging it to make one clip look better either does nothing principled
or quietly breaks that 5% guarantee for every other input. If the threshold should move,
it should move because it was recalibrated against better data (option C), not because
one result was uncomfortable.

---

## 3. Recommendation

- Do **B** now — it's strictly additive, safe, and makes the existing honest-but-generic
  caveat concrete. (Queued, not yet applied — the edit was interrupted for this writeup.)
- Treat **C/D** as the real fix, on your own timeline, since both need Kaggle and new
  real-video sourcing that only you can do. Say the word and I'll draft the recalibration
  script (C) or a data-collection + retrain spec (D) to take to Kaggle.
- Skip **A** (status quo is honest but under-explains this case) and avoid **E** entirely.

Tell me which of B/C/D (any combination) to act on.
