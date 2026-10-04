import os
from pathlib import Path

os.environ["HF_HOME"] = r"D:\HF_Models"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

AE_CONFIGS = {
    "sd1": {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "subfolder": "vae",
        "type": "kl",       # KL-regularized VAE
    },
    "sd2": {
        "repo_id": "sd2-community/stable-diffusion-2-1",
        "subfolder": "vae",
        "type": "kl",
    },
    "kandinsky": {
        "repo_id": "kandinsky-community/kandinsky-2-1",
        "subfolder": "movq",
        "type": "vq",       # vector-quantized VAE
    },
}

IMAGE_SIZE = 512
DEVICE = "cpu"
LPIPS_NET = "vgg"         # matches the paper's main setting

LPIPS_LAYER_INDEX = 1     # 0-based index into lpips retPerLayer
LPIPS_LAYER_NAME = 2      # paper's 1-based name, for labels/metadata only

DATA_DIR = _PROJECT_ROOT / "data"
RESULTS_DIR = _PROJECT_ROOT / "results"
MODELS_DIR = _PROJECT_ROOT / "models"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"