"""
data_utils.py - RAVDESS data loading and preprocessing for SincNet
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from collections import Counter

# We only use 4 emotions from the 8 available in RAVDESS
EMOTION_MAP = {"01": 0, "03": 1, "04": 2, "05": 3}
EMOTION_LABELS = ["neutral", "happy", "sad", "angry"]


def parse_ravdess_filename(filename):
    """Extract emotion code and speaker ID from RAVDESS filename format."""
    basename = os.path.splitext(filename)[0]
    parts = basename.split("-")
    if len(parts) != 7:
        return None
    return parts[2], int(parts[6])


def collect_file_paths(data_dir):
    """Walk through RAVDESS directory and collect all valid audio samples."""
    samples = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".wav"):
                continue
            parsed = parse_ravdess_filename(f)
            if parsed is None:
                continue
            emotion_code, speaker_id = parsed
            if emotion_code not in EMOTION_MAP:
                continue
            samples.append({
                "path": os.path.join(root, f),
                "emotion": EMOTION_MAP[emotion_code],
                "emotion_name": EMOTION_LABELS[EMOTION_MAP[emotion_code]],
                "speaker": speaker_id - 1,  # 0-indexed
            })
    return samples


def print_dataset_stats(samples):
    """Print basic dataset statistics."""
    print(f"Total samples: {len(samples)}")
    emotion_counts = Counter(s["emotion_name"] for s in samples)
    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")
    print(f"\nSpeakers: {len(set(s['speaker'] for s in samples))}")


def load_full_audio(file_path, target_sr=16000):
    """Load audio file, convert to mono and resample to target sample rate."""
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform.squeeze(0)


def load_and_preprocess_audio(file_path, target_sr=16000, max_length_sec=3.0):
    """Load audio with fixed length (for visualisation purposes)."""
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    max_samples = int(target_sr * max_length_sec)
    if waveform.shape[1] > max_samples:
        start = (waveform.shape[1] - max_samples) // 2
        waveform = waveform[:, start:start + max_samples]
    elif waveform.shape[1] < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[1]))
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


class RAVDESSDataset(Dataset):
    """
    RAVDESS dataset that supports two modes:
    
    Training (random_crop=True):
        Returns a random 200ms chunk from each audio file.
        This follows the original SincNet paper approach where each epoch
        sees different random crops, effectively augmenting the small dataset.
    
    Evaluation (random_crop=False):
        Returns the full audio padded/trimmed to fixed length.
        The model splits this into chunks internally.
    """
    def __init__(self, samples, target_sr=16000, chunk_size=3200,
                 max_length_sec=3.0, random_crop=False, amp_factor=0.2):
        self.samples = samples
        self.target_sr = target_sr
        self.chunk_size = chunk_size
        self.max_length_sec = max_length_sec
        self.random_crop = random_crop
        self.amp_factor = amp_factor
        self._cache = {}  # cache loaded audio in memory

    def _get_audio(self, path):
        if path not in self._cache:
            self._cache[path] = load_full_audio(path, self.target_sr)
        return self._cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        signal = self._get_audio(sample["path"])
        emotion = torch.tensor(sample["emotion"], dtype=torch.long)
        speaker = torch.tensor(sample["speaker"], dtype=torch.long)

        if self.random_crop:
            # Pick a random 200ms window from the audio
            snt_len = signal.shape[0]
            if snt_len > self.chunk_size:
                start = random.randint(0, snt_len - self.chunk_size - 1)
                chunk = signal[start:start + self.chunk_size]
            else:
                chunk = torch.nn.functional.pad(signal, (0, self.chunk_size - snt_len))

            # Random amplitude change (following original paper)
            amp = random.uniform(1.0 - self.amp_factor, 1.0 + self.amp_factor)
            chunk = chunk * amp

            # Normalise
            max_val = chunk.abs().max()
            if max_val > 0:
                chunk = chunk / max_val

            return chunk.unsqueeze(0), emotion, speaker

        else:
            # Return full audio at fixed length
            max_samples = int(self.target_sr * self.max_length_sec)
            if signal.shape[0] > max_samples:
                start = (signal.shape[0] - max_samples) // 2
                waveform = signal[start:start + max_samples]
            elif signal.shape[0] < max_samples:
                waveform = torch.nn.functional.pad(signal, (0, max_samples - signal.shape[0]))
            else:
                waveform = signal

            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

            return waveform.unsqueeze(0), emotion, speaker


def split_by_speaker(samples, val_speakers=[3, 4], test_speakers=[23, 24]):
    """Speaker-independent split: test speakers never appear in training."""
    val_set = {s - 1 for s in val_speakers}
    test_set = {s - 1 for s in test_speakers}
    train, val, test = [], [], []
    for s in samples:
        if s["speaker"] in test_set:
            test.append(s)
        elif s["speaker"] in val_set:
            val.append(s)
        else:
            train.append(s)
    print(f"Speaker-independent split: Train={len(train)} Val={len(val)} Test={len(test)}")
    return train, val, test


def split_random_per_speaker(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Random split where all speakers appear in train/val/test."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for speaker_id in range(24):
        speaker_files = [s for s in samples if s['speaker'] == speaker_id]
        rng.shuffle(speaker_files)
        n = len(speaker_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(speaker_files[:n_train])
        val.extend(speaker_files[n_train:n_train + n_val])
        test.extend(speaker_files[n_train + n_val:])
    print(f"Random split: Train={len(train)} Val={len(val)} Test={len(test)}")
    return train, val, test


def prepare_dataloaders(train_samples, val_samples, test_samples,
                        batch_size=128, chunk_size=3200,
                        max_length_sec=3.0, target_sr=16000):
    """Create train/val/test dataloaders. Training uses random crops."""
    train_ds = RAVDESSDataset(train_samples, target_sr=target_sr, chunk_size=chunk_size,
                               max_length_sec=max_length_sec, random_crop=True)
    val_ds = RAVDESSDataset(val_samples, target_sr=target_sr, chunk_size=chunk_size,
                             max_length_sec=max_length_sec, random_crop=False)
    test_ds = RAVDESSDataset(test_samples, target_sr=target_sr, chunk_size=chunk_size,
                              max_length_sec=max_length_sec, random_crop=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=min(32, len(val_samples)), shuffle=False),
        DataLoader(test_ds, batch_size=min(32, len(test_samples)), shuffle=False),
    )