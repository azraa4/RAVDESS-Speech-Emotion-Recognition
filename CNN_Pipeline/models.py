"""
models.py

Three models:
  - EmotionCNN: 2D CNN for emotion recognition from mel-spectrograms
  - SpeakerNet: 1D CNN + BiLSTM + Attention for speaker identification
  - MultiTaskNet: shared encoder (same as EmotionCNN) with two heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from augment_utils import SpecAugment


class EmotionCNN(nn.Module):
    """
    Deeper 2D CNN for emotion recognition from mel-spectrograms.
    5 conv blocks (32->64->128->256->256) with global average pooling.
    Extra depth helps capture higher-level temporal patterns.
    """
    def __init__(self, num_emotions=4, dropout_rate=0.1, use_specaugment=False):
        super().__init__()

        self.use_specaugment = use_specaugment
        if use_specaugment:
            self.spec_augment = SpecAugment(freq_mask_param=8, time_mask_param=16)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(dropout_rate)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(dropout_rate)

        # no pooling in block 5 — feature maps are already small enough
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout2d(dropout_rate)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 128)
        self.drop_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_emotions)

    def forward(self, x):
        if self.use_specaugment and self.training:
            x = self.spec_augment(x)

        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))

        x = self.global_pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)

    def extract_features(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        return self.global_pool(x).view(x.size(0), -1)


class Attention(nn.Module):
    # Attention learns to focus on the most informative time frames.
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output):
        scores = self.attention(lstm_output)
        weights = F.softmax(scores, dim=1)
        return torch.sum(weights * lstm_output, dim=1), weights


class SpeakerNet(nn.Module):
    """1D CNN + BiLSTM + Attention for speaker identification."""
    def __init__(self, n_mels=128, num_speakers=24, cnn_channels=64,
                 lstm_hidden=128, lstm_layers=2, dropout_rate=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_rate),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout_rate),
        )
        self.lstm = nn.LSTM(cnn_channels * 2, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout_rate if lstm_layers > 1 else 0)
        self.attention = Attention(lstm_hidden * 2)
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(lstm_hidden, num_speakers)

    def forward(self, x):
        x = x.squeeze(1)           # (B, n_mels, T)
        x = self.cnn(x)            # (B, C, T/4)
        x = x.permute(0, 2, 1)    # (B, T/4, C) for LSTM
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        x = F.relu(self.fc1(context))
        x = self.drop(x)
        return self.fc2(x)

    def extract_features(self, x):
        x = x.squeeze(1)
        x = self.cnn(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return context


class MultiTaskNet(nn.Module):
    """
    Multi-task model: shared CNN encoder (same as EmotionCNN) with two heads.

    The idea is that learning speaker identity as an auxiliary task might
    force the encoder to learn richer features (pitch, formants, speaking
    style) that also help with emotion recognition.

    Emotion head: global avg pool -> FC (same path as EmotionCNN)
    Speaker head: collapse frequency, keep time -> BiLSTM + Attention -> FC
    """
    def __init__(self, num_emotions=4, num_speakers=24, dropout_rate=0.1,
                 lstm_hidden=64):
        super().__init__()

        # Shared encoder — identical to EmotionCNN
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(dropout_rate)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(dropout_rate)

        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout2d(dropout_rate)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Emotion head same as EmotionCNN classifier
        self.emo_fc1 = nn.Linear(256, 128)
        self.emo_drop = nn.Dropout(dropout_rate)
        self.emo_fc2 = nn.Linear(128, num_emotions)

        # Speaker head BiLSTM on time-axis features
        self.spk_freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.spk_lstm = nn.LSTM(256, lstm_hidden, num_layers=1,
                                batch_first=True, bidirectional=True)
        self.spk_attn = Attention(lstm_hidden * 2)
        self.spk_fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.spk_drop = nn.Dropout(dropout_rate)
        self.spk_fc2 = nn.Linear(64, num_speakers)

    def _shared_encoder(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        return x  # (B, 256, H, W)

    def forward(self, x):
        feat = self._shared_encoder(x)

        # Emotion: global pool -> FC (same path as EmotionCNN)
        emo_feat = self.global_pool(feat).view(feat.size(0), -1)
        emo_out = F.relu(self.emo_fc1(emo_feat))
        emo_out = self.emo_drop(emo_out)
        emo_out = self.emo_fc2(emo_out)

        # Speaker: collapse freq, keep time -> BiLSTM -> Attention
        spk_feat = self.spk_freq_pool(feat).squeeze(2)  # (B, 256, T)
        spk_feat = spk_feat.permute(0, 2, 1)             # (B, T, 256)
        lstm_out, _ = self.spk_lstm(spk_feat)
        context, _ = self.spk_attn(lstm_out)
        spk_out = F.relu(self.spk_fc1(context))
        spk_out = self.spk_drop(spk_out)
        spk_out = self.spk_fc2(spk_out)

        return emo_out, spk_out
