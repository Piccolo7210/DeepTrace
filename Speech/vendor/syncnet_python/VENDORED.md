# Vendored: syncnet_python

- **Source**: https://github.com/joonson/syncnet_python
- **Pinned commit**: `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623` (2026-04-17)
- **Vendored on**: 2026-08-27
- **License**: MIT (Copyright Joon Son Chung) — see `LICENSE.md` in this directory,
  unmodified. Weight redistribution terms (`syncnet_v2.model`, `sfd_face.pth`) are not
  covered by this license and weren't independently verified — see Oxford VGG's own
  terms at https://www.robots.ox.ac.uk/~vgg/software/lipsync/ before distributing
  beyond personal/academic use.

## Why vendored instead of pip-installed

`speech_forensics_model.pt` (`Speech/models/`) was trained on `S(i,·)` sync features
produced by exactly this code + the official `syncnet_v2.model`/`sfd_face.pth`
checkpoints (see `Speech/kaggle/speechcode.ipynb`). Reimplementing S3FD/SyncNet from
scratch would risk subtle preprocessing/architecture mismatches against what the
checkpoint was actually trained on; there's no PyPI package for this repo.

## Patch applied (1 file)

`detectors/s3fd/__init__.py` — `PATH_WEIGHT` was a hardcoded, CWD-relative literal
(`'./detectors/s3fd/weights/sfd_face.pth'`). Since `run_pipeline.py` has no
`if __name__ == "__main__":` guard (module-level `argparse.parse_args()` plus
unconditional pipeline execution — see `Speech/syncnet_wrapper.py`'s docstring), it
must run as a subprocess, so the parent process can't monkeypatch this module
directly. Changed to read `os.environ.get('SFD_WEIGHTS_PATH', <original default>)`,
letting `Speech/syncnet_wrapper.py` point it at the single canonical copy in
`Speech/models/sfd_face.pth` via the subprocess's environment instead of duplicating
the 90MB weight file into this tree. The original default is preserved as a fallback
so this vendored copy still works standalone (matching upstream's own layout) if
someone drops weights into `detectors/s3fd/weights/` directly.

No other files were modified.
