"""
data_utils.py
Handles loading RAVDESS wav files and turning them into mel spectrograms.
I kept this in a separate file to keep the notebook cleaner.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from collections import Counter

# only using 4 emotions from the 8 available in RAVDESS
# dropped calm, fearful, disgust, surprised they felt too similar to each other
EMOTION_MAP = {"01": 0, "03": 1, "04": 2, "05": 3}
EMOTION_LABELS = ["neutral", "happy", "sad", "angry"]

# RAVDESS filenames encode metadata
def parse_ravdess_filename(filename):
    basename = os.path.splitext(filename)[0]
    parts = basename.split("-")
    if len(parts) != 7:
        return None
    return parts[2], int(parts[6])

# Walk through all the actor folders and grab the wav files we need.
def collect_file_paths(data_dir):
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
                "speaker": speaker_id - 1,  # making it 0-indexed
            })
    return samples

def print_dataset_stats(samples):
    print(f"Total samples: {len(samples)}")
    emotion_counts = Counter(s["emotion_name"] for s in samples)
    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")
    print(f"\nSpeakers: {len(set(s['speaker'] for s in samples))}")

# To load a wav file, convert to mono, resample to 16kHz, and fix the length.
def load_and_preprocess_audio(file_path, target_sr=16000, max_length_sec=3.0):
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
    return waveform

# To convert audio to log-mel spectrogram.
def waveform_to_mel_spectrogram(waveform, sr=16000, n_mels=64, n_fft=1024, hop_length=512):
    mel_spec = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)(waveform)
    return torch.log(mel_spec + 1e-9)

# Returns (mel_spectrogram, emotion_label, speaker_label) per sample.
class RAVDESSDataset(Dataset):
    def __init__(self, samples, target_sr=16000, max_length_sec=3.0,
                 n_mels=64, n_fft=1024, hop_length=512, augment=False):
        self.samples = samples
        self.target_sr = target_sr
        self.max_length_sec = max_length_sec
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform = load_and_preprocess_audio(sample["path"], self.target_sr, self.max_length_sec)
        if self.augment:
            from augment_utils import augment_waveform
            waveform = augment_waveform(waveform, sr=self.target_sr)
        mel_spec = waveform_to_mel_spectrogram(waveform, self.target_sr, self.n_mels, self.n_fft, self.hop_length)
        emotion = torch.tensor(sample["emotion"], dtype=torch.long)
        speaker = torch.tensor(sample["speaker"], dtype=torch.long)
        return mel_spec, emotion, speaker

# Speaker-independent split hold out entire speakers for val and test.      
def split_by_speaker(samples, val_speakers=[3, 4], test_speakers=[23, 24]):
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

def prepare_dataloaders(train_samples, val_samples, test_samples,
                        batch_size=16, augment_train=False, **kwargs):
    train_ds = RAVDESSDataset(train_samples, augment=augment_train, **kwargs)
    val_ds = RAVDESSDataset(val_samples, augment=False, **kwargs)
    test_ds = RAVDESSDataset(test_samples, augment=False, **kwargs)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))
