"""Constants for the AIGVDet reproduction (Bai, Lin, Cao - arXiv 2403.16638).

Two blocks:

* INFERENCE CONTRACT - values the trained networks depend on. Changing any of
  them invalidates existing checkpoints and the app's preprocessing.
* DATA, TRAINING, STORAGE - values that follow the official code or the
  implementation plan. Change them only with a documented reason.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Inference contract
# --------------------------------------------------------------------------- #
CROP_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FUSION_ALPHA = 0.5
DECISION_THRESHOLD = 0.5  # official code default; the paper text states 0.1

BRANCH_RGB = "rgb"
BRANCH_FLOW = "flow"
BRANCHES = (BRANCH_RGB, BRANCH_FLOW)

LABEL_REAL = 0
LABEL_FAKE = 1
CLASS_DIRS = {"0_real": LABEL_REAL, "1_fake": LABEL_FAKE}

# --------------------------------------------------------------------------- #
# Training data (Hugging Face release, verified in the plan, section 3.1)
# --------------------------------------------------------------------------- #
HF_REPO_ID = "xmulz/AIGVDet"
HF_REVISION = "c1326befed8ea265e547984e1e25fe83e6b2a7bf"
HF_ARCHIVES = {
    ("train", BRANCH_RGB): "train/moonvalley_vos2_original_crop.zip",
    ("train", BRANCH_FLOW): "train/moonvalley_vos2_crop_op.zip",
    ("val", BRANCH_RGB): "val/moonvalley_vos2_original_crop.zip",
    ("val", BRANCH_FLOW): "val/moonvalley_vos2_crop_op.zip",
}
EXPECTED_VIDEOS_PER_CLASS = {"train": 500, "val": 50}
EXPECTED_FRAMES_PER_VIDEO = {BRANCH_RGB: 95, BRANCH_FLOW: 93}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# --------------------------------------------------------------------------- #
# Augmentation (official ``datasets.py`` / ``config.py``)
# --------------------------------------------------------------------------- #
AUG_BLUR_PROB = 0.1
AUG_BLUR_SIGMA = 0.5
AUG_JPEG_PROB = 0.1
AUG_JPEG_QUALITY = 75
AUG_FLIP_PROB = 0.5

# --------------------------------------------------------------------------- #
# Optimisation (paper section III-A, official ``trainer.py`` / ``train.py``)
# --------------------------------------------------------------------------- #
SEED = 3407
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
FC_INIT_STD = 0.02
LR_DECAY_FACTOR = 10.0
LR_FLOOR = 1e-6
LR_PATIENCE = 5
MIN_DELTA_FIRST = 0.001
MIN_DELTA_AFTER_DROP = 0.002
MAX_EPOCHS = 40
BUDGET_FRAMES_PER_VIDEO = 32

# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
DRIVE_ROOT = "gdrive:SPL3/Video-AIGVDet"
DRIVE_MANIFESTS = f"{DRIVE_ROOT}/manifests"
DRIVE_CKPTS = f"{DRIVE_ROOT}/ckpts"
SCRATCH_CANDIDATES = ("/kaggle/temp", "/tmp")
MIN_FREE_SCRATCH_GB = 35

RAFT_UPDATES = 20
MAX_VIDEO_FRAMES = 95
PAIRS_PER_VIDEO = 32
COND_RGB_JPEG_QUALITY = 75  # conditioning variant for RGB frames
COND_FLOW_JPEG_QUALITY = 95  # conditioning variant for flow maps (cv2.imwrite default)
RESOLUTION_BUCKET_EDGES = (448, 720)
MAX_FLOW_PIXELS = 1_500_000
