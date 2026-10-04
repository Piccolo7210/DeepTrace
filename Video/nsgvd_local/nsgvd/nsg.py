"""Normalized Spatiotemporal Gradient (NSG) -- exact port of the official NSG-VD estimator.

    score  = -UNet(2x - 1, t * 999)[:, :3] / std_VPSDE(t),  t = (diffuse_steps + 1) / 1000 = 0.006
             (evaluated on the CLEAN frame -- the official code adds no noise)
    dx_t   = (x_{t+1} - x_{t-1}) / 2, one-sided at the first / last frame
    denom_t= <score_t, dx_t> + lambda   (sum over C, H, W);   NSG_t = score_t / denom_t
We store score (fp16) + denom (fp32): same information, but immune to fp16 overflow when denom ~ 0.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from nsgvd import config as C
from nsgvd.utils import get_logger

# 256x256_diffusion_uncond flags (NSG-VD libs/eps_ad/imagenet.yml + guided-diffusion defaults)
UNET_KWARGS = dict(
    image_size=256, num_channels=256, num_res_blocks=2, channel_mult="", learn_sigma=True,
    class_cond=False, use_checkpoint=False, attention_resolutions="32,16,8", num_heads=4,
    num_head_channels=64, num_heads_upsample=-1, use_scale_shift_norm=True, dropout=0.0,
    resblock_updown=True, use_new_attention_order=False,
)


def ensure_guided_diffusion_importable() -> None:
    path = str(C.GUIDED_DIFFUSION_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def build_unet(use_fp16: bool, **overrides) -> torch.nn.Module:
    ensure_guided_diffusion_importable()
    from guided_diffusion.script_util import create_model

    return create_model(**{**UNET_KWARGS, "use_fp16": use_fp16, **overrides})


def load_unet(ckpt_path, device: torch.device, use_fp16: bool | None = None) -> torch.nn.Module:
    """Frozen score network, prepared exactly like the official RevGuidedDiffusion."""
    use_fp16 = C.USE_FP16_UNET if use_fp16 is None else use_fp16
    use_fp16 = bool(use_fp16 and device.type == "cuda")      # fp16 convs are for GPUs only
    model = build_unet(use_fp16)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, TypeError, ValueError):             # legacy (non-zip) checkpoint format
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    if use_fp16:
        model.convert_to_fp16()
    return model.eval().requires_grad_(False).to(device)


def t_value(diffuse_steps: int | None = None) -> float:
    return ((C.DIFFUSE_STEPS if diffuse_steps is None else diffuse_steps) + 1) / 1000.0


def vpsde_log_mean_coeff(t: torch.Tensor) -> torch.Tensor:
    bmin, bmax = C.VPSDE_BETA_MIN, C.VPSDE_BETA_MAX
    return -0.25 * t ** 2 * (bmax - bmin) - 0.5 * t * bmin


def vpsde_std(t: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1.0 - torch.exp(2.0 * vpsde_log_mean_coeff(t)))


def central_difference(x: torch.Tensor) -> torch.Tensor:
    """(B, T, ...) -> same shape; official `_get_score_velocity` pixel difference (Delta-t = 1)."""
    dx = torch.empty_like(x)
    dx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
    dx[:, 0] = x[:, 1] - x[:, 0]
    dx[:, -1] = x[:, -1] - x[:, -2]
    return dx


def nsg_from_score(score: torch.Tensor, denom: torch.Tensor,
                   boundary_scale: float | None = None) -> torch.Tensor:
    """Rebuild NSG (..., T, 3, H, W) float32 from stored score + denominator (..., T)."""
    boundary_scale = C.BOUNDARY_FRAME_SCALE if boundary_scale is None else boundary_scale
    nsg = score.float() / denom.float()[..., None, None, None]
    if boundary_scale != 1.0:
        nsg[..., 0, :, :, :] *= boundary_scale
        nsg[..., -1, :, :, :] *= boundary_scale
    return nsg


class NSGEstimator:
    """Batched score + NSG-denominator computation on one device."""

    def __init__(self, unet: torch.nn.Module, device: torch.device, micro_batch: int | None = None, logger=None):
        self.unet, self.device = unet, device
        self.size, self.res = C.FEATURE_SIZE, C.SCORE_RESOLUTION
        self.t = t_value()
        self.lam = C.NSG_LAMBDA
        self.micro_batch = micro_batch or C.K2_FRAMES_PER_FORWARD
        self.log = logger or get_logger()

    @torch.no_grad()
    def score_model_input(self, x_m11: torch.Tensor) -> torch.Tensor:
        """Score at model resolution for an input already in [-1, 1] (score_sde get_score_fn)."""
        t = torch.full((x_m11.shape[0],), self.t, device=x_m11.device, dtype=torch.float32)
        out = self.unet(x_m11, t * 999)
        eps = out[:, : out.shape[1] // 2].float()          # learn_sigma: first 3 of 6 channels
        return -eps / vpsde_std(t)[:, None, None, None]

    @torch.no_grad()
    def score(self, x01: torch.Tensor) -> torch.Tensor:
        """(N, 3, S, S) frames in [0, 1] -> score at feature size S."""
        inp = x01
        if self.res != x01.shape[-1]:
            inp = F.interpolate(x01, size=(self.res, self.res), mode="bilinear", align_corners=False)
        score = self.score_model_input(2 * inp - 1)
        if score.shape[-1] != x01.shape[-1]:
            score = F.interpolate(score, size=tuple(x01.shape[-2:]), mode="bilinear", align_corners=False)
        return score

    def _score_all(self, flat: torch.Tensor) -> torch.Tensor:
        outs, i = [], 0
        while i < flat.shape[0]:
            mb = self.micro_batch
            try:
                outs.append(self.score(flat[i:i + mb]))
                i += mb
            except torch.cuda.OutOfMemoryError:
                if mb == 1:
                    raise
                torch.cuda.empty_cache()
                self.micro_batch = max(1, mb // 2)
                self.log.warning(f"CUDA OOM -> UNet micro-batch reduced to {self.micro_batch} frames")
        return torch.cat(outs)

    @torch.no_grad()
    def score_and_denom(self, frames_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, 3, S, S) uint8 -> score (B, T, 3, S, S) fp32, denom (B, T) fp32."""
        x = frames_u8.to(self.device, non_blocking=True).float().div_(255.0)
        b, t = x.shape[:2]
        score = self._score_all(x.reshape(b * t, *x.shape[2:])).reshape(x.shape)
        denom = (score * central_difference(x)).sum(dim=(2, 3, 4)) + self.lam
        return score, denom


@torch.no_grad()
def tweedie_check(est: NSGEstimator, x01: torch.Tensor, seed: int = 0) -> dict:
    """Noise clean frames to t, denoise with the score (Tweedie). Right sign/scale => error drops."""
    x01 = x01.to(est.device).float()
    if x01.shape[-1] != est.res:
        x01 = F.interpolate(x01, size=(est.res, est.res), mode="bilinear", align_corners=False)
    x0 = 2 * x01 - 1
    t = torch.full((x0.shape[0],), est.t, device=x0.device)
    alpha = torch.exp(vpsde_log_mean_coeff(t))[:, None, None, None]
    std = vpsde_std(t)[:, None, None, None]
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    eps = torch.randn(x0.shape, generator=gen, device=x0.device)
    x_t = alpha * x0 + std * eps
    score = est.score_model_input(x_t)
    x0_hat = (x_t + std ** 2 * score) / alpha
    mse_noisy = float(((x_t / alpha - x0) ** 2).mean())
    mse_denoised = float(((x0_hat - x0) ** 2).mean())
    cos = float(F.cosine_similarity((-score * std).flatten(), eps.flatten(), dim=0))
    return {"mse_noisy": mse_noisy, "mse_denoised": mse_denoised,
            "improvement_x": mse_noisy / max(mse_denoised, 1e-12), "eps_cosine": cos,
            "passed": bool(mse_denoised < mse_noisy / 1.3 and cos > 0.3)}


def temporal_derivative(denom: torch.Tensor) -> torch.Tensor:
    """d/dt log p = -<score, dx>/dt (Prop. 1), from the stored denominator (dt = 1)."""
    return -(denom - C.NSG_LAMBDA)


def fraction_near_zero(values: torch.Tensor, bound: float = 0.1) -> float:
    values = values.flatten().float()
    return float((values.abs() <= bound).float().mean()) if values.numel() else math.nan
