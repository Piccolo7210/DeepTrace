import torch

from config import AE_CONFIGS, DEVICE
from diffusers import AutoencoderKL, VQModel


def load_autoencoders():
    aes = {}
    for name, cfg in AE_CONFIGS.items():
        print(f"Loading {name} from {cfg['repo_id']} ...")
        try:
            if cfg["type"] == "kl":
                model = AutoencoderKL.from_pretrained(
                    cfg["repo_id"],
                    subfolder=cfg["subfolder"],
                    torch_dtype=torch.float32,
                )
            elif cfg["type"] == "vq":
                model = VQModel.from_pretrained(
                    cfg["repo_id"],
                    subfolder=cfg["subfolder"],
                    torch_dtype=torch.float32,
                )
            else:
                raise ValueError(f"Unknown AE type: {cfg['type']}")

            model.to(DEVICE)
            model.eval()
            aes[name] = model
            print(f"  -> {name} loaded ({sum(p.numel() for p in model.parameters()):,} params)")
        except Exception as e:
            print(f"  -> WARNING: could not load {name}: {e}")
    return aes