# Curating real images for calibration

`src/calibrate.py` compares images in this folder against the generated images in
`data/generated/*/` to pick a detection threshold. The quality of that threshold depends
directly on how representative this real-image set is — a sloppy set gives a biased
threshold. This guide covers how to source and select ~50-100 images for here.

## Why it matters

The paper shows (Sec. 5.4) that reconstruction error correlates with image complexity:
simple/flat real images (plain backgrounds, logos, screenshots) reconstruct deceptively
well and can look "fake" to the detector, while complex real images (textures, clutter,
fine detail) reconstruct poorly and look clearly "real". If this folder skews toward
simple images, the calibrated threshold will be biased toward flagging real photos as
generated. Aim for a realistic spread of complexity, not an "easy mode" set that's all
plain portraits or clean product shots.

## Sourcing

Use one of these, in order of preference:

1. **Your own photos** — phone/camera photos are ideal: guaranteed real, no licensing
   questions.
2. **Public-domain / CC0 sources** — Unsplash, Pexels, Pixabay, Wikimedia Commons. Pick
   images explicitly marked public domain or CC0, not just "free to view."

Avoid anything you're not sure is genuinely camera-captured. No AI-generated images, no
heavily stylized/filtered images, no screenshots of digital art or renders.

## Selection checklist

- [ ] **Real, unedited-ish** — camera photos, not AI art, not heavy filters/stylization.
- [ ] **Resolution**: smaller side is already ≥512px. `src/reconstruct.py` resizes every
  image to 512×512 (`IMAGE_SIZE` in `src/config.py`) — feeding it an undersized image just
  means it gets blurred by the resize, which distorts the LPIPS distance for reasons
  unrelated to authenticity. Bigger-than-512 is fine (gets center-cropped/resized down).
- [ ] **Content variety**: mix of subjects — portraits, nature, objects, cluttered scenes,
  some simple ones too — roughly mirroring the kind of content the generated set covers
  (SD/Kandinsky outputs tend to be people, objects, scenes, art-style compositions). Don't
  make every image a macro shot of the same subject.
- [ ] **No duplicates/near-duplicates** — don't include multiple takes of the same shot;
  it inflates the apparent sample size without adding real signal.
- [ ] **Clean source** — avoid images with heavy pre-existing JPEG artifacts, watermarks,
  or borders; these add reconstruction error that has nothing to do with real-vs-generated.

## Target count

Aim for **~50-100 images**, roughly matching the size of the generated subset you're
pulling per model from the Zenodo release — a balanced set gives a more reliable
threshold and AP estimate.

## Where they go

Drop images directly into this folder (`data/real/`), flat — no subfolders.
`calibrate.py` scans this directory non-recursively. `example.jpg` (already here) can
stay; it counts as one of the samples.

## Before running calibration

Once you've assembled the set, do a quick sanity pass:
- Count the files — should be in the ~50-100 range.
- Skim thumbnails for anything that snuck in that shouldn't be there (near-duplicates,
  screenshots, anything not obviously a real photo).

Then run:

```
python src/calibrate.py
```
