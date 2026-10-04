"""
Autoregressive anomaly detection model (paper Sec. 3.2, Eq. 3-5, Sec. 4.1): a
decoder-only causal Transformer that predicts each frame's time-delay distribution
from the previous frames' distributions. Trained only on real video (train_ar.py);
at test time (calibrate.py, detect.py) its per-frame prediction loss is the anomaly
score — high loss means the video's synchronization behaves unlike anything in the
real training distribution.

This is a direct, literal implementation of the paper's stated spec (2 decoder
blocks, 16 heads, 256 channels, soft-CE against the delay distribution — the
best-performing variant in Table 3), unlike the sync model's encoders, whose exact
architecture the paper only cites from elsewhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AR_DIM, AR_DROPOUT, AR_HEADS, AR_LAYERS, AR_MAX_LEN, DELAY_BINS


class ARModel(nn.Module):
    def __init__(self, input_dim=DELAY_BINS, dim=AR_DIM, layers=AR_LAYERS,
                 heads=AR_HEADS, dropout=AR_DROPOUT, max_len=AR_MAX_LEN,
                 activation="relu"):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout,
            activation=activation, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out_proj = nn.Linear(dim, input_dim)

    def forward(self, x):
        """x: (B, N, input_dim) sequence of delay distributions x_1..x_N. Returns
        (B, N, input_dim) raw logits; position i's logits are the prediction of x_{i+1}
        given x_1..x_i (causal masking), so callers shift by one (see step_losses)."""
        B, N, _ = x.shape
        h = self.in_proj(x) + self.pos_embed[:, :N]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(N).to(x.device)
        h = self.decoder(h, mask=causal_mask, is_causal=True)
        return self.out_proj(h)


def step_losses(model, seq):
    """Per-timestep soft cross-entropy (paper Eq. 5): predicts x_{i+1} from x_1..x_i
    for every i. seq: (B, N, DELAY_BINS) with N>=2. Returns (B, N-1) losses, one per
    predicted step; index t is the loss for predicting seq[:, t+1] from seq[:, :t+1]."""
    logits = model(seq)[:, :-1]          # (B, N-1, D): position t predicts seq[t+1]
    targets = seq[:, 1:]                  # (B, N-1, D)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1)


def sequence_anomaly_score(model, seq):
    """Cumulative prediction loss over a whole clip (paper Eq. 4) — the fake-score.
    seq: (N, DELAY_BINS) for a single clip (no batch dim). Returns a scalar float and
    the per-frame loss curve (N-1,) for plotting/temporal localization."""
    with torch.no_grad():
        losses = step_losses(model, seq.unsqueeze(0)).squeeze(0)  # (N-1,)
    return float(losses.sum()), losses


def load_pretrained_ar_model(checkpoint_path, device="cpu", max_len=None):
    """Loads Speech/models/speech_forensics_model.pt -- the Kaggle-trained checkpoint
    for the pretrained-SyncNet pipeline (see Speech/syncnet_wrapper.py) -- into this
    module's ARModel. Its AutoregressiveAnomalyModel class is architecturally
    identical given its own config, but three things differ from a bare
    ARModel(**cfg) and must be bridged or the model silently computes wrong outputs
    (or fails to load at all):

      1. state-dict key `pos_embedding` (checkpoint) vs `pos_embed` (this class) --
         renamed below before load_state_dict.
      2. activation="gelu" (checkpoint's TransformerEncoderLayer) vs this class's
         "relu" default -- same weight shapes, wrong math if not passed through.
      3. max_len=512 (checkpoint's default at export time) vs this file's AR_MAX_LEN
         (256, for the from-scratch pipeline) -- NOT recorded in the checkpoint's own
         "config" dict, so it must be supplied by the caller (see
         Speech/config.py's PRETRAINED_AR_MAX_LEN) or the pos_embed shape won't match.

    Returns (model, ckpt) -- ckpt still has "thresholds"/"seq_len"/"stride"/etc. that
    callers need for scoring, so it's returned alongside rather than discarded.
    """
    if max_len is None:
        from config import PRETRAINED_AR_MAX_LEN
        max_len = PRETRAINED_AR_MAX_LEN

    # Not a pure state_dict (it's model_state_dict + config + thresholds + metadata),
    # so weights_only=True would raise -- this checkpoint is a trusted, project-owned
    # artifact (this project's own Kaggle export), not an untrusted download.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = ARModel(
        input_dim=cfg["feat_dim"], dim=cfg["d_model"], layers=cfg["n_layers"],
        heads=cfg["n_heads"], dropout=cfg.get("dropout", AR_DROPOUT),
        max_len=max_len, activation="gelu",
    ).to(device)

    state_dict = dict(ckpt["model_state_dict"])
    if "pos_embedding" in state_dict:
        state_dict["pos_embed"] = state_dict.pop("pos_embedding")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, ckpt
