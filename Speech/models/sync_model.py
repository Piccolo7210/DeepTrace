"""
Audio-visual synchronization model (paper Sec. 3.1, Sec. 4.1): a visual encoder, an
audio encoder, and a small Transformer fusion head that scores how well a video
window and an audio window co-occur. Softmaxing that score over a window of
candidate audio delays gives S(i,·) (Eq. 1) — the feature the AR anomaly model
(models/ar_model.py) is trained on.

Needs nothing beyond torch/torchvision (the log-mel spectrogram is computed with
torch.stft + a hand-built mel filterbank instead of torchaudio), so this file — and
everything that imports only this file, e.g. train_sync.py and smoke_test.py — runs
on a plain CPU install with no extra packages beyond what's already in the repo's
requirements.txt. torchvision is already a repo dependency (used by the AEROBLADE
image module) and ships on Kaggle's default image, so pulling in a real video
backbone from it costs nothing extra.

Visual encoder: paper Sec. 4.1 says "Following Chen et al. [18], we use ResNet-18
2D+3D [42, 43] as the visual encoder" (Hara et al.'s 3D-CNN study + He et al.'s
ResNet), following a hybrid-conv recipe from Tran et al., "A Closer Look at
Spatiotemporal Convolutions for Action Recognition" (CVPR 2018) — the paper that
actually coined "MC3" (early 3D convs, later 2D convs) as a ResNet-18-based
2D+3D hybrid. torchvision.models.video.mc3_18 is that architecture; that's what's
used below (weights=None — trained from scratch here, same as the paper's own
recipe on LRS2/LRS3).

Audio encoder: VGG-M (Chatfield et al., "Return of the devil in the details", 2014)
isn't in torchvision, so it's reimplemented directly below, matching its 5-conv-block
channel/kernel progression (96-256-512-512-512, kernels 7-5-3-3-3) with strides
adapted so an 80x21 mel-spectrogram (a tiny fraction of VGG-M's native 224x224 image
input) doesn't collapse to nothing; BatchNorm replaces the original's local response
normalization, the standard modern swap.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision

from config import (
    AUDIO_SR,
    EMBED_DIM,
    FUSION_HEADS,
    FUSION_LAYERS,
    MEL_HOP_LENGTH,
    MEL_N_FFT,
    MEL_N_MELS,
    MEL_WIN_LENGTH,
)


def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Standard HTK-scale triangular mel filterbank, (n_mels, n_fft//2+1)."""
    fmax = fmax or sr / 2

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bin_pts = np.floor((n_fft + 1) * mel_to_hz(mel_pts) / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=AUDIO_SR, n_fft=MEL_N_FFT, win_length=MEL_WIN_LENGTH,
                 hop_length=MEL_HOP_LENGTH, n_mels=MEL_N_MELS):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.register_buffer("filterbank", torch.from_numpy(_mel_filterbank(sample_rate, n_fft, n_mels)))

    def forward(self, wav):  # wav: (B, samples) -> (B, n_mels, time)
        spec = torch.stft(
            wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, return_complex=True,
        )
        power = spec.real ** 2 + spec.imag ** 2  # (B, freq, time)
        mel = torch.einsum("mf,bft->bmt", self.filterbank, power)
        return torch.log(mel + 1e-6)


class VisualEncoder(nn.Module):
    """5-frame RGB clip -> embed_dim vector, via torchvision's MC3-18 (ResNet-18
    with early 3D / later 2D residual blocks — the "ResNet-18 2D+3D" the paper cites,
    Tran et al. CVPR 2018), trained from scratch (weights=None)."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = torchvision.models.video.mc3_18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embed_dim)

    def forward(self, clip):  # clip: (B, T, 3, H, W) -> torchvision video convention (B, 3, T, H, W)
        return self.backbone(clip.permute(0, 2, 1, 3, 4))


class AudioEncoder(nn.Module):
    """0.2s raw waveform -> embed_dim vector, via a log-mel spectrogram and a
    VGG-M-style 2D CNN (paper Sec. 4.1): 5 conv blocks with VGG-M's channel/kernel
    progression (96-256-512-512-512, kernels 7-5-3-3-3), strides chosen so the
    80x21 mel-spectrogram survives to a non-empty spatial map (see module docstring)."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.melspec = LogMelSpectrogram()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, wav):  # wav: (B, samples)
        mel = self.melspec(wav).unsqueeze(1)  # (B, 1, n_mels, time)
        x = self.pool(self.net(mel)).flatten(1)
        return self.fc(x)


class FusionHead(nn.Module):
    """Late-fusion module h(v, a) (paper Sec. 3.1): a small Transformer encoder over
    the 2-token [visual, audio] sequence, pooled and projected to a scalar phi(V, A)."""

    def __init__(self, dim=EMBED_DIM, layers=FUSION_LAYERS, heads=FUSION_HEADS):
        super().__init__()
        self.modality_embed = nn.Parameter(torch.zeros(2, dim))
        nn.init.normal_(self.modality_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(dim, 1)

    def _fuse(self, v, a):  # v, a: (M, dim) -> (M,)
        tokens = torch.stack([v + self.modality_embed[0], a + self.modality_embed[1]], dim=1)  # (M, 2, dim)
        out = self.encoder(tokens).mean(dim=1)  # (M, dim)
        return self.head(out).squeeze(-1)

    def pairwise(self, v_embed, a_embed):
        """All-pairs phi: v_embed (Nv, d), a_embed (Na, d) -> logits (Nv, Na)."""
        Nv, Na, d = v_embed.size(0), a_embed.size(0), v_embed.size(1)
        v_exp = v_embed.unsqueeze(1).expand(Nv, Na, d).reshape(Nv * Na, d)
        a_exp = a_embed.unsqueeze(0).expand(Nv, Na, d).reshape(Nv * Na, d)
        return self._fuse(v_exp, a_exp).view(Nv, Na)

    def paired(self, v_embed, a_embed_seq):
        """Per-item phi against K candidates each: v_embed (B, d), a_embed_seq
        (B, K, d) -> logits (B, K). Used for the delay window (paper Eq. 1)."""
        B, K, d = a_embed_seq.shape
        v_exp = v_embed.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
        a_exp = a_embed_seq.reshape(B * K, d)
        return self._fuse(v_exp, a_exp).view(B, K)


class SyncModel(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.visual = VisualEncoder(embed_dim)
        self.audio = AudioEncoder(embed_dim)
        self.fusion = FusionHead(embed_dim)

    def encode_audio(self, wav):
        """wav: (B, samples) for a single candidate each, or (B, K, samples) for K
        candidates each (e.g. the delay window) -> (B, d) or (B, K, d)."""
        if wav.dim() == 3:
            B, K, S = wav.shape
            return self.audio(wav.reshape(B * K, S)).view(B, K, -1)
        return self.audio(wav)

    def cross_video_logits(self, video_clip, audio_clip):
        """In-batch contrastive logits (Nv=Na=B): target is the diagonal (item i's
        audio truly matches item i's video). 'Cross-video negatives' curriculum phase."""
        v = self.visual(video_clip)
        a = self.encode_audio(audio_clip)
        return self.fusion.pairwise(v, a)

    def window_logits(self, video_clip, audio_window):
        """audio_window: (B, DELAY_BINS, samples), candidates from the SAME video at
        every delay in [-tau, tau]. Softmax over dim=1 is S(i,·) (Eq. 1). 'Within-video
        negatives' curriculum phase, and also what extract_features.py calls at
        inference time."""
        v = self.visual(video_clip)
        a = self.encode_audio(audio_window)
        return self.fusion.paired(v, a)
