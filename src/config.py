import os
os.environ["HF_HOME"] = r"D:\HF_Models"

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

IMAGE_SIZE = 512          # resize images to this before encoding
DEVICE = "cpu"
LPIPS_NET = "vgg"         # matches the paper's main setting

# LPIPS2 is the best-performing single layer per Tab. 1 of the paper. The lpips
# package's retPerLayer list is 0-based (index 0 = slice1/relu1_2 = the paper's
# LPIPS1), so the paper's LPIPS2 is index 1. Keeping the index and the display
# name as separate constants avoids indexing with the label (which computed
# LPIPS3 by mistake) or labelling with the index.
LPIPS_LAYER_INDEX = 1     # 0-based index into lpips retPerLayer
LPIPS_LAYER_NAME = 2      # paper's 1-based name, for labels/metadata only

DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"
THRESHOLD_PATH = "models/threshold.json"