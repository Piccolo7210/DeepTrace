"""NSG-VD Kaggle pipeline -- the ONLY file you normally edit.

Every module reads settings as ``C.NAME`` at call time (``from nsgvd import config as C``),
so after editing this cell just re-run it and then the "Bootstrap" cell (it reloads modules).
Paths honour environment variables so the same code can be tested outside Kaggle.
"""
from __future__ import annotations

import os
from pathlib import Path

# ════════════════════════════ 1. GOOGLE DRIVE (rclone) ════════════════════════════
RCLONE_REMOTE = "gdrive"            # the [name] of the Drive remote inside your rclone.conf
DRIVE_ROOT = "SPL3/Video-NSGVD"     # project folder inside that remote (plan §5.1)
VIDEOS_DIR = "videos_subset"        # source videos, relative to DRIVE_ROOT
WILD_REALS_DIR = "wild_reals"       # optional folder of your own real clips (phone/YouTube); "" = off

# ════════════════════════════ 2. DATASET (plan §4.2) ════════════════════════════
# `folders` are fnmatch patterns (case-insensitive) relative to VIDEOS_DIR, matched against the
# layout confirmed on Drive: T2V/<gen>_mp4, I2V/<gen>_mp4 (unpacked GVD archives) and
# real/kinetics400, real/msrvtt (copied from the Kaggle datasets).
# dev = videos for train/val/ref, test = videos for the final test.
# Real side follows NSG-VD's own protocol: Kinetics-400 for training, MSR-VTT for testing.
# Not used: I2V/pika_mp4, I2V/neverends_mp4 (its 3 duplicate files with T2V/neverends_mp4 cannot leak).
SOURCES = {
    "kinetics400":    dict(label=0, modality="-",   folders=["real/kinetics400"],    hints=["kinetics"],   dev=3000, test=0),
    "msrvtt":         dict(label=0, modality="-",   folders=["real/msrvtt"],         hints=["msrvtt"],     dev=1000, test=1000),
    "moonvalley_t2v": dict(label=1, modality="T2V", folders=["T2V/moonvalley_mp4"],  hints=["moonvalley"], dev=1200, test=0),
    "videocraft":     dict(label=1, modality="T2V", folders=["T2V/videocraft_mp4"],  hints=["videocraft"], dev=800,  test=0),
    "pika_t2v":       dict(label=1, modality="T2V", folders=["T2V/pika_mp4"],        hints=["pika"],       dev=700,  test=0),
    "neverends_t2v":  dict(label=1, modality="T2V", folders=["T2V/neverends_mp4"],   hints=["neverends"],  dev=700,  test=0),
    "emu":            dict(label=1, modality="T2V", folders=["T2V/emu_mp4"],         hints=["emu"],        dev=500,  test=0),
    "videopoet":      dict(label=1, modality="T2V", folders=["T2V/videopoet_mp4"],   hints=["videopoet"],  dev=100,  test=0),
    "hotshot":        dict(label=1, modality="T2V", folders=["T2V/hotshot_mp4"],     hints=["hotshot"],    dev=0,    test=500),
    "moonvalley_i2v": dict(label=1, modality="I2V", folders=["I2V/moonvalley_mp4"],  hints=["moonvalley"], dev=0,    test=452),
    "sora":           dict(label=1, modality="T2V", folders=["T2V/sora_mp4"],        hints=["sora"],       dev=0,    test=48),
}
REAL_DEV_SPLIT = {"train": 3000, "ref": 500, "val": 500}   # must add up to the real `dev` total (3000 + 1000)
FAKE_VAL_TOTAL = 500          # fakes carved from the fake dev pool into val, proportional per generator
WILD_REF_FRACTION = 0.5       # wild reals: this share joins the reference pool, the rest -> split "wild"
SPLIT_SEED = 42

# Optional: YouTube-VOS2 reals from HuggingFace (PNG frame folders) for reference-pool diversity.
HF_YVOS2_COUNT = 0            # 0 = off; e.g. 150 adds 150 reals to the reference pool
HF_REPO_ID = "xmulz/AIGVDet"
HF_YVOS2_PREFIX = "train/trainset_1/0_real/"

# ════════════════════════════ 3. FRAMES + NSG (official NSG-VD settings) ════════════════════════════
T_FRAMES = 8                  # frames per clip
FRAME_WINDOW = 95            # canonical sampler window: first min(N, 95) frames (plan §4.3b)
FEATURE_SIZE = 224            # frames / NSG maps are 224x224 (official configs/nsg-vd-224x224)
SCORE_RESOLUTION = 224        # resolution the diffusion UNet runs at (official 224; 128 = "fast mode")
WATERMARK_BOTTOM_FRAC = 0.10  # bottom band removed from EVERY video of BOTH classes (verify in K1-1)
DIFFUSE_STEPS = 5             # official `data.diffuse_steps`; continuous time t = (steps + 1) / 1000
NSG_LAMBDA = 1e-10            # official epsilon in the NSG denominator
BOUNDARY_FRAME_SCALE = 100.0  # official loader multiplies the NSG of the first/last frame by 100
VPSDE_BETA_MIN = 0.1          # continuous VP-SDE used by the official score function
VPSDE_BETA_MAX = 20.0
USE_FP16_UNET = True          # official: guided-diffusion convert_to_fp16 (T4 has fast fp16)
K2_FRAMES_PER_FORWARD = 24    # UNet micro-batch in frames = official 3 clips/call; halves on CUDA OOM

# Robustness variants (plan §4.3a / §8 gate 6). Applied identically to both classes.
TRAIN_AUG_VARIANTS = ["jpeg", "rescale"]   # extra NSG copies of the TRAIN split
TEST_AUG_VARIANTS = ["jpeg"]               # re-compressed copy of the TEST split ([] to skip)
JPEG_QUALITY_RANGE = (60, 95)
RESCALE_FACTOR = 0.6

# ════════════════════════════ 4. CHECKPOINTING / STORAGE ════════════════════════════
SHARD_SIZE = 100              # videos per checkpointed batch ("shard") in K1 and K2
LOCAL_CACHE_BUDGET_GB = 4.0  # max shard data kept in /kaggle/working (Drive always keeps everything)
SCRATCH_BUDGET_GB = 40.0      # max NSG data that K3-K5 pull into scratch (Kaggle scratch is ~60 GB)
RCLONE_TRANSFERS = 8

# ════════════════════════════ 5. KERNEL TRAINING (K3, official hyper-parameters) ════════════════════════════
RUN_NAME = "res224_mp_gvd6"   # change to start a separate training run (e.g. an ablation)
MMD_MP = True                 # True = MMD-MP (model.is_yy_zero=True), False = MMD-D
BATCH_SIZE = 24               # real AND fake clips per step, per GPU
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.999)
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 30      # epochs without val-AUROC improvement
VAL_EVERY = 1
KERNEL_SIGMA = 1000.0         # official model.sigma   (raw-feature bandwidth, pre-parameterisation)
KERNEL_SIGMA0 = 0.1           # official model.sigma0  (deep-feature bandwidth)
KERNEL_EPSILON = 10.0         # official model.epsilon (init: ep ~ 1e-10)
FEATURE_DIM = 300
BALANCE_GENERATORS = True     # sample fake generators uniformly (plan K3)
TRAIN_VARIANTS = ["clean", "jpeg", "rescale"]   # NSG variants K3 trains on ("clean" is required)
NUM_WORKERS = 2               # DataLoader workers per GPU process (Kaggle has 4 CPU cores)
SEED = 42
KEEP_EPOCH_CKPT_EVERY = 10

# ════════════════════════════ 6. REFERENCE / CALIBRATION / EVALUATION (K4, K5) ════════════════════════════
N_REF = 100                   # paper default reference-set size
REF_SOURCE_MIX = {"msrvtt": 0.5, "kinetics400": 0.3, "wild": 0.2}   # renormalised over available sources
TARGET_FPR = 0.05
TAU_PAPER = 1.0
TEST_TIERS = {
    "moonvalley_i2v": "Easiest: seen generator, unseen modality",
    "hotshot": "Unseen generator (short, low-resolution clips)",
    "sora": "Hardest: unseen, near-photorealistic (small n -> see CI)",
}
BOOTSTRAP_ROUNDS = 1000

# ════════════════════════════ 7. PINNED THIRD-PARTY ARTEFACTS ════════════════════════════
GUIDED_DIFFUSION_REPO = "https://github.com/openai/guided-diffusion.git"
GUIDED_DIFFUSION_COMMIT = "22e0df8183507e13a7813f8d38d51b072ca1e67c"
UNET_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
UNET_FILENAME = "256x256_diffusion_uncond.pt"
RCLONE_VERSION = "v1.74.4"
RCLONE_SHA256 = "fe435e0c36228e7c2f116a8701f01127bb1f694005fc11d1f27186c8bca4115d"
NSGVD_REPO_RAW = "https://raw.githubusercontent.com/ZSHsh98/NSG-VD"
NSGVD_REPO_COMMIT = "9a60f57fcb464b418899110a4717c8c8fd7687bc"
AUTHORS_BASELINE_CKPT = "standard-SEINE-mp.pth"

# ════════════════════════════ 8. PATHS (normally leave alone) ════════════════════════════
WORK_DIR = Path(os.environ.get("NSGVD_WORK", "/kaggle/working"))
SCRATCH_DIR = Path(os.environ.get("NSGVD_SCRATCH", "/kaggle/temp/nsgvd"))
CODE_ROOT = WORK_DIR / "nsgvd_code"
LOCAL_CKPT_DIR = WORK_DIR / "nsgvd_ckpt"        # local mirror of the Drive tree (fast access)
THIRD_PARTY_DIR = WORK_DIR / "third_party"
GUIDED_DIFFUSION_DIR = THIRD_PARTY_DIR / "guided-diffusion"
RCLONE_BIN = os.environ.get("NSGVD_RCLONE_BIN", str(THIRD_PARTY_DIR / "bin" / "rclone"))
RCLONE_CONFIG = os.environ.get("RCLONE_CONFIG", "/root/.config/rclone/rclone.conf")
UNET_LOCAL_PATH = SCRATCH_DIR / "models" / UNET_FILENAME

# Drive / local-mirror layout (relative to DRIVE_ROOT and LOCAL_CKPT_DIR)
D_MANIFESTS = "manifests"
D_LEDGERS = "ledgers"
D_FRAMES =  f"frames/r{FEATURE_SIZE}"
D_NSG = f"nsg/steps{DIFFUSE_STEPS}_res{SCORE_RESOLUTION}"
D_CKPTS = "ckpts"
D_REFERENCE = "reference"
D_RESULTS = "results"
D_EXPORT = "export"
D_MODELS = "models"
MANIFEST_NAME = "manifest_v1.csv"

SPLITS_TO_PROCESS = ["ref", "val", "train", "test", "wild"]  # processing order in K1/K2
