import lpips
import torch
from config import DEVICE, LPIPS_NET, LPIPS_LAYER_INDEX, LPIPS_LAYER_NAME

# Load once and reuse — loading LPIPS' VGG16 backbone repeatedly is wasteful.
# spatial=True keeps the per-layer maps (upsampled to input H, W) instead of
# spatially averaging them, so a single model serves both the scalar
# reconstruction error and the pixel-wise heatmap.
_lpips_model = lpips.LPIPS(net=LPIPS_NET, spatial=True).to(DEVICE)
_lpips_model.eval()


def _layer_maps(x, x_tilde):
    """
    Runs the patched-spatial LPIPS forward pass and returns the per-layer
    error maps, each already upsampled to (1, 1, H, W).
    """
    with torch.no_grad():
        _, layer_maps = _lpips_model(x, x_tilde, retPerLayer=True, normalize=False)
    return layer_maps


def reconstruction_error(x, x_tilde, layer=LPIPS_LAYER_INDEX):
    """
    Computes the AE reconstruction error Delta_AE(x) = d(x, x_tilde) using a
    single LPIPS layer (the paper's LPIPS2 by default, its best-performing
    setting). `layer` is a 0-based index into the lpips retPerLayer list, not
    the paper's 1-based name. x and x_tilde must be (1, 3, H, W) tensors in
    [-1, 1] range.

    Returns a single float distance value.
    """
    layer_map = _layer_maps(x, x_tilde)[layer]
    return layer_map.mean().item()


def reconstruction_heatmap(x, x_tilde, layer=LPIPS_LAYER_INDEX):
    """
    Computes the pixel-wise reconstruction error map (no spatial averaging),
    used to visualize which regions drive the reconstruction error. `layer` is
    a 0-based index into the lpips retPerLayer list, not the paper's 1-based
    name. x and x_tilde must be (1, 3, H, W) tensors in [-1, 1] range.

    Returns a (H, W) numpy array.
    """
    layer_map = _layer_maps(x, x_tilde)[layer]
    return layer_map.squeeze(0).squeeze(0).cpu().numpy()


if __name__ == "__main__":
    from autoencoders import load_autoencoders
    from reconstruct import load_image_tensor, reconstruct
    from config import AE_CONFIGS

    aes = load_autoencoders()
    test_image_path = "data/real/example.jpg"
    x = load_image_tensor(test_image_path)

    for name, vae in aes.items():
        ae_type = AE_CONFIGS[name]["type"]
        x_tilde = reconstruct(vae, x, ae_type=ae_type)
        d = reconstruction_error(x, x_tilde)
        print(f"{name}: LPIPS{LPIPS_LAYER_NAME} distance = {d:.4f}")
