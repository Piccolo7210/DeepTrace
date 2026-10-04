"""
Shared paths and hyperparameters for the audio-visual anomaly detection module
(reproduction of Feng, Chen & Owens, "Self-Supervised Video Forensics by Audio-Visual
Anomaly Detection", CVPR 2023 — see `Speech/audio + video.pdf`), trained on the GRID
corpus (https://spandh.dcs.shef.ac.uk/gridcorpus/) instead of the paper's LRS2/LRS3.

This module is self-contained under Speech/, with its own data/models/results dirs,
mirroring how src/config.py roots the image-forensics (AEROBLADE) module.
"""

from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parent

DATA_DIR = _MODULE_ROOT / "data"
GRID_RAW_DIR = DATA_DIR / "grid_raw"          # user-downloaded GRID zips, extracted here
PROCESSED_DIR = DATA_DIR / "grid_processed"    # face crops (.npy) + 16kHz wav per utterance
FEATURES_DIR = DATA_DIR / "av_features"        # extracted S(i,:) sequences per utterance
MANIFEST_PATH = DATA_DIR / "manifest.csv"
SYNTHETIC_FAKES_DIR = DATA_DIR / "synthetic_fakes"

MODELS_DIR = _MODULE_ROOT / "models"
SYNC_CKPT_DIR = MODELS_DIR / "sync"
AR_CKPT_DIR = MODELS_DIR / "ar"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

RESULTS_DIR = _MODULE_ROOT / "results"

DEVICE = "cpu"  # overridden to "cuda" automatically when available; see utils.get_device()

# --- Video / audio I/O (paper Sec. 4.1, matched to GRID's native format) -------------
FPS = 25                    # GRID is natively 25fps, same as the paper's visual encoder input
AUDIO_SR = 16000             # paper's audio encoder rate; GRID ships 25kHz, resampled down
CROP_SIZE = 112              # square face-crop side, in pixels
CROP_MARGIN = 1.6            # multiply the detected face bbox side by this before cropping

# --- Synchronization features (paper Sec. 3.1-3.2) ------------------------------------
TAU = 15                     # max time delay in frames considered on each side
DELAY_BINS = 2 * TAU + 1     # 31 possible discrete delays -> distribution length

# --- Sync model architecture (paper Sec. 4.1) ------------------------------------------
VISUAL_CLIP_LEN = 5           # frames per video window fed to the visual encoder
AUDIO_CLIP_SEC = 0.2          # seconds of audio fed to the audio encoder per window
MEL_N_MELS = 80
MEL_HOP_LENGTH = 160
MEL_WIN_LENGTH = 320
MEL_N_FFT = 320
EMBED_DIM = 512               # visual/audio embedding dim before fusion
FUSION_LAYERS = 3
FUSION_HEADS = 4

# --- Sync model training -----------------------------------------------------------
SYNC_LR = 1e-4
SYNC_BATCH_PHASE1 = 16        # cross-video negatives
SYNC_BATCH_PHASE2 = 40        # within-video negatives

# --- AR anomaly model architecture (paper Sec. 4.1) -------------------------------------
AR_DIM = 256
AR_LAYERS = 2
AR_HEADS = 16
AR_DROPOUT = 0.1
SEQUENCE_LENGTH = 50           # frames per training sequence (2.0s at 25fps)
AR_MAX_LEN = 256               # positional-encoding capacity; covers training windows
                                # (50) and full GRID clips (~75 frames at 25fps) alike

# --- AR model training ---------------------------------------------------------------
AR_LR = 1e-3
AR_WEIGHT_DECAY = 1e-6
AR_BATCH_SIZE = 16
AR_WARMUP_STEPS = 500

# --- Speaker split (mirrors paper's "divide into 3 splits", Sec. 4.2) ------------------
# Populated by manifest.py from --speakers; kept here only as the default group ratio.
SPLIT_NAMES = ("sync_train", "ar_train", "heldout_real")

# --- Pretrained-SyncNet pipeline (the actually-deployed detector) ----------------------
# Unlike everything above (the from-scratch reproduction attempt in train_sync.py /
# models/sync_model.py, never completed -- no checkpoint exists), this is the pipeline
# that actually backs the app's "Speech Forensics" screen: see syncnet_wrapper.py,
# detect_pretrained.py, and models/ar_model.py's load_pretrained_ar_model(). It swaps
# the custom sync encoder for the official pretrained SyncNet + S3FD (vendored below),
# and reuses this file's TAU/DELAY_BINS/FPS/AUDIO_SR/MODELS_DIR as-is.
PRETRAINED_MODEL_PATH = MODELS_DIR / "speech_forensics_model.pt"
SYNCNET_CKPT_PATH = MODELS_DIR / "syncnet_v2.model"
SFD_CKPT_PATH = MODELS_DIR / "sfd_face.pth"
SYNCNET_VENDOR_DIR = _MODULE_ROOT / "vendor" / "syncnet_python"

# The notebook that trained speech_forensics_model.pt built its AR model with this
# max_len default; the checkpoint's own "config" dict does NOT record it, so it must
# be supplied explicitly at load time or the positional-embedding shape won't match.
PRETRAINED_AR_MAX_LEN = 512

# run_pipeline.py --min_track / --min_face_size, inherited unmodified from the
# training notebook (GRID-tuned: short low-res clips). If face-track extraction fails
# on a real, higher-resolution user upload, loosen these first.
MIN_TRACK_FRAMES = 25
MIN_FACE_SIZE_PX = 40
