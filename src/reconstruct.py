import torch
from PIL import Image
from torchvision import transforms
from config import IMAGE_SIZE, DEVICE, AE_CONFIGS

_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),          # [0, 1] range, shape (C, H, W)
])


def load_image_tensor(path):
    """Load an image file and return a (1, 3, H, W) tensor in [-1, 1] range."""
    img = Image.open(path).convert("RGB")
    tensor = _preprocess(img)               # [0, 1]
    tensor = tensor * 2 - 1                  # [-1, 1], what the VAE expects
    return tensor.unsqueeze(0).to(DEVICE)    # add batch dim


def reconstruct(vae, image_tensor, ae_type="kl"):
    """
    Runs image_tensor through encoder -> decoder.
    Branches on ae_type since AutoencoderKL and VQModel have different APIs.
    Returns the reconstruction as a (1, 3, H, W) tensor in [-1, 1] range.
    """
    with torch.no_grad():
        if ae_type == "kl":
            latents = vae.encode(image_tensor).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            recon = vae.decode(latents / vae.config.scaling_factor).sample

        elif ae_type == "vq":
            # VQModel: encode() returns latents directly, no distribution,
            # no scaling_factor step. decode() takes those latents straight.
            encoded = vae.encode(image_tensor)
            latents = encoded.latents
            recon = vae.decode(latents).sample

        else:
            raise ValueError(f"Unknown ae_type: {ae_type}")

    return recon


def tensor_to_pil(tensor):
    """Convert a (1, 3, H, W) tensor in [-1, 1] back to a viewable PIL image."""
    tensor = (tensor.clamp(-1, 1) + 1) / 2   # back to [0, 1]
    tensor = tensor.squeeze(0).cpu()
    return transforms.ToPILImage()(tensor)


if __name__ == "__main__":
    from autoencoders import load_autoencoders

    aes = load_autoencoders()
    test_image_path = "data/real/example.jpg"

    x = load_image_tensor(test_image_path)
    for name, vae in aes.items():
        ae_type = AE_CONFIGS[name]["type"]     # "kl" or "vq"
        x_tilde = reconstruct(vae, x, ae_type=ae_type)
        out = tensor_to_pil(x_tilde)
        out.save(f"results/recon_{name}.png")
        print(f"Saved reconstruction using {name} -> results/recon_{name}.png")