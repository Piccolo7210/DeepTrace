"""Constants for the video forensics module (NSG-VD video detection).

Module-level constants, no dataclass — the same shape as src/config.py and
Speech/config.py, so all three modules read the same way.

The values under "inference contract" are NOT tunable: they describe how the
detector's features were computed on Kaggle (NSGVD_KAGGLE_HANDOFF.md §7).
Changing any of them makes the local features disagree with the trained model,
which shows up as silently wrong probabilities rather than an error. The
canonical values live in Video/nsgvd_local/nsgvd/config.py (the Kaggle bundle,
kept unmodified); the ones repeated here are only those the app layer needs.
"""

from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_ROOT.parent

# ---- paths -----------------------------------------------------------------
BUNDLE_DIR = _MODULE_ROOT / "nsgvd_local"          # unmodified Kaggle bundle
BUNDLE_MODELS_DIR = BUNDLE_DIR / "models"
GUIDED_DIFFUSION_DIR = BUNDLE_DIR / "third_party" / "guided-diffusion"
MODELS_DIR = _MODULE_ROOT / "models"               # our sklearn-free tree dump
RESULTS_DIR = PROJECT_ROOT / "results"

UNET_PATH = BUNDLE_MODELS_DIR / "256x256_diffusion_uncond.pt"
UNET_SHA256_PREFIX = "a37c32fffd31"                # handoff §9 acceptance check
# Exact size of the official checkpoint. Checked before torch.load because a
# half-downloaded file is still a file: without this, an interrupted download
# surfaces as "PytorchStreamReader failed ... checkpoint file is corrupted",
# which reads like a broken model rather than an unfinished transfer.
UNET_BYTES = 2_211_383_297
JOBLIB_BUNDLE_PATH = BUNDLE_MODELS_DIR / "nsg_stats_gbm.joblib"
TREE_DUMP_PATH = MODELS_DIR / "nsg_gbm_trees.npz"
PARITY_SAMPLES_PATH = BUNDLE_MODELS_DIR / "parity_samples.npz"

DEVICE = "cpu"  # no CUDA on this machine; torch.cuda is still used automatically if present

# ---- inference contract (do not change) ------------------------------------
T_FRAMES = 8                  # frames scored per clip
FRAME_WINDOW = 95             # clip is sampled from the first min(N, 95) frames
FEATURE_SIZE = 224            # frames and NSG maps are 224x224
WATERMARK_BOTTOM_FRAC = 0.10  # bottom band dropped from every video, both classes
UNET_MICRO_BATCH = 8          # frames per UNet forward; lower to 4 on a low-RAM machine

# ---- presentation ----------------------------------------------------------
# The threshold itself comes from the trained bundle (tau = 0.6915, calibrated for
# 5% false positives on validation reals), never from this file.
UNCERTAIN_BAND = 0.10         # |p_fake - tau| < this -> "uncertain" (handoff §8.6)

# Carried in every result and in detect_nsgvd.py's JSON output; the app does not
# display them. These are measured limitations from the Kaggle
# evaluation (handoff §3), not boilerplate: the detector saw 2023-24 generators
# only, and part of its accuracy comes from resolution/compression differences
# between the real and fake halves of the training data.
CAVEATS = [
    "Trained on AI video generators from 2023-24 (Moonvalley, VideoCraft, Pika, "
    "NeverEnds, Emu, VideoPoet). Newer generators may not be detected.",
    "Test AUROC on unseen generators was 0.84, and only 41% of generated clips "
    "were caught at this threshold — a 'real' verdict is weak evidence.",
    "In the training data, real and generated clips also differed in resolution "
    "and compression, so part of the accuracy may come from those cues rather "
    "than from the physics of the video.",
]

SUPPORTED_EXTS = (
    "*.mp4", "*.mov", "*.m4v", "*.webm", "*.mkv", "*.avi",
    "*.mpg", "*.mpeg", "*.wmv", "*.flv", "*.3gp", "*.ts", "*.gif",
)
