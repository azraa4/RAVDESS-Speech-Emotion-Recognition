"""
sincnet.py
SincNet models for the coursework. Based on the paper by Ravanelli & Bengio (2018).
 
Instead of mel-spectrograms, SincNet works directly on raw audio waveforms.
The first conv layer uses parametrised sinc functions as bandpass filters —
only the cutoff frequencies are learned, which makes it very parameter-efficient.
 
I adapted the original speaker recognition architecture for emotion recognition
by keeping the encoder the same but adjusting the FC head sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# --- LayerNorm (from the original SincNet repo) ---

class LayerNorm(nn.Module):
    # LayerNorm that works on the feature dimension.
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# --- SincConv layer (learnable bandpass filters) ---

class SincConv1d(nn.Module):
    """
    Sinc-based convolution layer that learns bandpass filter parameters.
    Only two parameters per filter (low freq and bandwidth) are learned,
    which makes it much more parameter-efficient than standard Conv1d.
    Follows the SincConv_fast implementation from the original paper repo.
    """
    def __init__(self, out_channels=80, kernel_size=251, sample_rate=16000,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

        # Initialise filter frequencies on mel scale
        low_hz = 30.0
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel_low = 2595.0 * np.log10(1.0 + low_hz / 700.0)
        mel_high = 2595.0 * np.log10(1.0 + high_hz / 700.0)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # Learnable parameters: low cutoff frequency and bandwidth
        self.low_hz_ = nn.Parameter(torch.Tensor(hz_points[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz_points)).view(-1, 1))

        # Hamming window (computed for right half only, then mirrored)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int(self.kernel_size / 2))
        self.register_buffer('window_', 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size))

        # Time indices for the left half of the filter
        n = (self.kernel_size - 1) / 2.0
        self.register_buffer('n_', 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate)

    def forward(self, x):
        self.n_ = self.n_.to(x.device)
        self.window_ = self.window_.to(x.device)

        # To compute valid frequency boundaries
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2.0)
        band = (high - low)[:, 0]

        # Build bandpass filters using sinc functions
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
                          / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=0)

# --- SincNet CNN encoder ---

class SincEncoder(nn.Module):
    """
    Three-layer CNN with SincConv as first layer.
    
    Architecture (following the paper):
        SincConv(80, 251) -> |abs| -> MaxPool(3) -> LayerNorm -> LeakyReLU
        Conv1d(80, 60, 5)           -> MaxPool(3) -> LayerNorm -> LeakyReLU
        Conv1d(60, 60, 5)           -> MaxPool(3) -> LayerNorm -> LeakyReLU
        -> Flatten
    
    No padding (valid convolution) — this is how the original repo does it.
    Takes 200ms chunks (3200 samples at 16kHz) and outputs a flat vector.
    """
    def __init__(self, sinc_channels=80, sinc_kernel=251,
                 sample_rate=16000, chunk_size=3200):
        super().__init__()
        self.sinc = SincConv1d(out_channels=sinc_channels, kernel_size=sinc_kernel,
                               sample_rate=sample_rate)

        # Calculate output sizes after each layer (no padding)
        s1 = (chunk_size - sinc_kernel + 1) // 3
        s2 = (s1 - 5 + 1) // 3
        s3 = (s2 - 5 + 1) // 3

        self.ln1 = LayerNorm([sinc_channels, s1])
        self.conv2 = nn.Conv1d(sinc_channels, 60, kernel_size=5, padding=0)
        self.ln2 = LayerNorm([60, s2])
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5, padding=0)
        self.ln3 = LayerNorm([60, s3])

        self.out_dim = 60 * s3  # 6420

    def forward(self, x):
        # First layer: SincConv + absolute value (as per paper)
        x = torch.abs(self.sinc(x))
        x = F.max_pool1d(x, 3)
        x = F.leaky_relu(self.ln1(x))

        x = self.conv2(x)
        x = F.max_pool1d(x, 3)
        x = F.leaky_relu(self.ln2(x))

        x = self.conv3(x)
        x = F.max_pool1d(x, 3)
        x = F.leaky_relu(self.ln3(x))

        return x.view(x.size(0), -1)  # flatten


# =============================================
# Helper: split audio into chunks and classify
# =============================================

def _chunk_and_avg(model_classify_fn, x, chunk_size):
    """
    Split full audio into non-overlapping chunks, classify each one,
    and return the average logits. Used at evaluation time.
    """
    batch_size = x.size(0)
    starts = list(range(0, x.size(2) - chunk_size + 1, chunk_size))
    num_chunks = len(starts)

    chunks = torch.stack([x[:, 0, s:s + chunk_size] for s in starts], dim=1)
    chunks_flat = chunks.reshape(batch_size * num_chunks, 1, chunk_size)

    logits = model_classify_fn(chunks_flat)
    logits = logits.view(batch_size, num_chunks, -1)
    return logits.mean(dim=1)


# =============================================
# Model 1: Emotion Recognition
# =============================================

class EmotionSincNet(nn.Module):
    """
    SincNet-based emotion classifier.
    
    Training: classifies single random 200ms chunks
    Evaluation: classifies all chunks from full audio, averages predictions
    
    Architecture: SincEncoder -> LayerNorm -> FC(256) -> FC(256) -> FC(4)
    """
    def __init__(self, num_emotions=4, fc_hidden=256,
                 dropout_rate=0.2, sample_rate=16000, chunk_size=3200, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size

        self.encoder = SincEncoder(sample_rate=sample_rate, chunk_size=chunk_size)
        enc_dim = self.encoder.out_dim

        self.ln_inp = LayerNorm(enc_dim)
        self.fc1 = nn.Linear(enc_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden, momentum=0.05)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.bn2 = nn.BatchNorm1d(fc_hidden, momentum=0.05)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(fc_hidden, num_emotions)

    def _classify(self, x):
        """Classify a single chunk."""
        x = self.encoder(x)
        x = self.ln_inp(x)
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x))))
        return self.fc_out(x)

    def forward(self, x):
        if x.size(2) <= self.chunk_size:
            return self._classify(x)  # training: single chunk
        else:
            return _chunk_and_avg(self._classify, x, self.chunk_size)  # eval: average chunks

    def extract_features(self, x):
        """Get intermediate features (for feature fusion experiments)."""
        if x.size(2) <= self.chunk_size:
            x = self.encoder(x)
            x = self.ln_inp(x)
            x = F.leaky_relu(self.bn1(self.fc1(x)))
            return F.leaky_relu(self.bn2(self.fc2(x)))
        else:
            batch_size = x.size(0)
            starts = list(range(0, x.size(2) - self.chunk_size + 1, self.chunk_size))
            chunks = torch.stack([x[:, 0, s:s+self.chunk_size] for s in starts], dim=1)
            flat = chunks.reshape(batch_size * len(starts), 1, self.chunk_size)
            feat = self.encoder(flat)
            feat = self.ln_inp(feat)
            feat = F.leaky_relu(self.bn1(self.fc1(feat)))
            feat = F.leaky_relu(self.bn2(self.fc2(feat)))
            return feat.view(batch_size, len(starts), -1).mean(dim=1)


# =============================================
# Model 2: Speaker Identification
# =============================================

class SpeakerSincNet(nn.Module):
    """
    SincNet-based speaker identifier.
    
    Same approach as EmotionSincNet but with larger FC layers (512 vs 256)
    since speaker identity requires more capacity to distinguish 24 speakers.
    
    Architecture: SincEncoder -> LayerNorm -> FC(512) -> FC(512) -> FC(24)
    """
    def __init__(self, num_speakers=24, fc_hidden=512,
                 dropout_rate=0.2, sample_rate=16000, chunk_size=3200):
        super().__init__()
        self.chunk_size = chunk_size

        self.encoder = SincEncoder(sample_rate=sample_rate, chunk_size=chunk_size)
        enc_dim = self.encoder.out_dim

        self.ln_inp = LayerNorm(enc_dim)
        self.fc1 = nn.Linear(enc_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden, momentum=0.05)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.bn2 = nn.BatchNorm1d(fc_hidden, momentum=0.05)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(fc_hidden, num_speakers)

    def _classify(self, x):
        x = self.encoder(x)
        x = self.ln_inp(x)
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x))))
        return self.fc_out(x)

    def forward(self, x):
        if x.size(2) <= self.chunk_size:
            return self._classify(x)
        else:
            return _chunk_and_avg(self._classify, x, self.chunk_size)

    def extract_features(self, x):
        if x.size(2) <= self.chunk_size:
            x = self.encoder(x)
            x = self.ln_inp(x)
            x = F.leaky_relu(self.bn1(self.fc1(x)))
            return F.leaky_relu(self.bn2(self.fc2(x)))
        else:
            batch_size = x.size(0)
            starts = list(range(0, x.size(2) - self.chunk_size + 1, self.chunk_size))
            chunks = torch.stack([x[:, 0, s:s+self.chunk_size] for s in starts], dim=1)
            flat = chunks.reshape(batch_size * len(starts), 1, self.chunk_size)
            feat = self.encoder(flat)
            feat = self.ln_inp(feat)
            feat = F.leaky_relu(self.bn1(self.fc1(feat)))
            feat = F.leaky_relu(self.bn2(self.fc2(feat)))
            return feat.view(batch_size, len(starts), -1).mean(dim=1)


# =============================================
# Combined: Multi-task model
# =============================================

class MultiTaskSincNet(nn.Module):
    """
    Shared SincNet encoder with two task-specific heads.
    
    The idea is that speaker and emotion information share low-level
    acoustic features, so a shared encoder can benefit both tasks.
    The speaker head acts as a regulariser for the emotion task.
    
    Emotion head: LayerNorm -> FC(256) -> FC(256) -> 4 classes
    Speaker head: LayerNorm -> FC(512) -> FC(512) -> N classes
    """
    def __init__(self, num_emotions=4, num_speakers=24,
                 emo_fc=256, spk_fc=512,
                 dropout_rate=0.2, sample_rate=16000, chunk_size=3200, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size

        # Shared encoder
        self.encoder = SincEncoder(sample_rate=sample_rate, chunk_size=chunk_size)
        enc_dim = self.encoder.out_dim

        # Emotion head
        self.emo_ln = LayerNorm(enc_dim)
        self.emo_fc1 = nn.Linear(enc_dim, emo_fc)
        self.emo_bn1 = nn.BatchNorm1d(emo_fc, momentum=0.05)
        self.emo_drop1 = nn.Dropout(dropout_rate)
        self.emo_fc2 = nn.Linear(emo_fc, emo_fc)
        self.emo_bn2 = nn.BatchNorm1d(emo_fc, momentum=0.05)
        self.emo_drop2 = nn.Dropout(dropout_rate)
        self.emo_out = nn.Linear(emo_fc, num_emotions)

        # Speaker head
        self.spk_ln = LayerNorm(enc_dim)
        self.spk_fc1 = nn.Linear(enc_dim, spk_fc)
        self.spk_bn1 = nn.BatchNorm1d(spk_fc, momentum=0.05)
        self.spk_drop1 = nn.Dropout(dropout_rate)
        self.spk_fc2 = nn.Linear(spk_fc, spk_fc)
        self.spk_bn2 = nn.BatchNorm1d(spk_fc, momentum=0.05)
        self.spk_drop2 = nn.Dropout(dropout_rate)
        self.spk_out = nn.Linear(spk_fc, num_speakers)

    def _emo_classify(self, enc):
        x = self.emo_ln(enc)
        x = self.emo_drop1(F.leaky_relu(self.emo_bn1(self.emo_fc1(x))))
        x = self.emo_drop2(F.leaky_relu(self.emo_bn2(self.emo_fc2(x))))
        return self.emo_out(x)

    def _spk_classify(self, enc):
        x = self.spk_ln(enc)
        x = self.spk_drop1(F.leaky_relu(self.spk_bn1(self.spk_fc1(x))))
        x = self.spk_drop2(F.leaky_relu(self.spk_bn2(self.spk_fc2(x))))
        return self.spk_out(x)

    def forward(self, x):
        if x.size(2) <= self.chunk_size:
            # Training: single chunk
            enc = self.encoder(x)
            return self._emo_classify(enc), self._spk_classify(enc)
        else:
            # Evaluation: average over chunks
            batch_size = x.size(0)
            starts = list(range(0, x.size(2) - self.chunk_size + 1, self.chunk_size))
            num_chunks = len(starts)

            chunks = torch.stack([x[:, 0, s:s+self.chunk_size] for s in starts], dim=1)
            flat = chunks.reshape(batch_size * num_chunks, 1, self.chunk_size)

            enc = self.encoder(flat)

            emo_logits = self._emo_classify(enc).view(batch_size, num_chunks, -1)
            spk_logits = self._spk_classify(enc).view(batch_size, num_chunks, -1)

            return emo_logits.mean(dim=1), spk_logits.mean(dim=1)