"""
augment_utils.py
Data augmentation: SpecAugment, mixup, noise injection, and class weights.
Tried different augmentations but mixup ended up being the only one
that consistently helped on this small dataset.
"""

import torch
import torch.nn as nn
import random
from collections import Counter


# --- Waveform augmentation ---

# To adds random gaussian noise to the audio.
def add_white_noise(waveform, noise_level=0.005):
    return waveform + torch.randn_like(waveform) * noise_level


# To apply waveform augmentation based on mode.
def augment_waveform(waveform, sr=16000, mode="noise", p=0.5):
    waveform = waveform.clone()
    if mode == "none":
        return waveform
    if mode in ("noise", "all"):
        if random.random() < p:
            waveform = add_white_noise(waveform, random.uniform(0.002, 0.008))
    return waveform


# --- Spectrogram augmentation ---

# SpecAugment: Masks random frequency bands and time steps in the spectrogram.
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=8, time_mask_param=16,
                 n_freq_masks=1, n_time_masks=1):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        if not self.training:
            return x
        x = x.clone()
        _, _, n_mels, time_steps = x.shape
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            x[:, :, f0:f0 + f, :] = 0
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, time_steps - t))
            x[:, :, :, t0:t0 + t] = 0
        return x


# --- Mixup augmentation ---

# Mixup: blend two random samples and their labels. Returns mixed inputs, pairs of targets, and the mixing coefficient.
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Compute loss for mixup: weighted combination of losses for both targets.
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# --- Class weights --- 

# Class weights for imbalanced data
def compute_class_weights(samples, num_classes=4):
    counts = Counter(s["emotion"] for s in samples)
    total = len(samples)
    return torch.FloatTensor([total / (num_classes * counts.get(i, 1)) for i in range(num_classes)])
