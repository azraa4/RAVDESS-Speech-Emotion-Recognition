# CNN vs SincNet for Speech Emotion Recognition and Speaker Identification

Comparing mel-spectrogram CNNs and raw-waveform [SincNet](https://arxiv.org/abs/1808.00158) for speech emotion recognition and speaker identification on RAVDESS. Both pipelines also have multi-task variants with a shared encoder and joint loss L = L_emotion + λ·L_speaker. Everything implemented from scratch in PyTorch, except the `SincConv` layer adapted from the [official repo](https://github.com/mravanelli/SincNet).

## Results

| Model | Task | Acc | F1 |
|---|---|---:|---:|
| EmotionCNN | Emotion | 58.9% | 0.599 |
| SpeakerNet | Speaker | 84.2% | 0.831 |
| MultiTask CNN (λ=0.1) | Emotion | 55.4% | 0.533 |
| EmotionSincNet | Emotion | 50.0% | 0.498 |
| SpeakerSincNet | Speaker | 93.3% | 0.930 |
| MultiTask SincNet (λ=0.1) | Emotion | 62.5% | 0.602 |

The interesting part: multi-task learning hurts the CNN (-3.6 pts) but helps SincNet a lot (+12.5 pts), making multi-task SincNet the best emotion model overall. The speaker task seems to push the sinc filters toward pitch/formant regions that also help with emotion — something the mel-spectrogram already provides for free.

## Dataset

[RAVDESS](https://zenodo.org/record/1188976) — 672 samples (neutral, happy, sad, angry) from 24 actors. Emotion uses a speaker-independent test split (unseen voices).

## Repo Structure

```
CNN_Pipeline/          EmotionCNN, SpeakerNet, MultiTaskNet + training notebook
SincNet_Pipeline/      SincNet models + training notebook
Case_Study/            single-sample inference demo across all models
```

## References

- Ravanelli & Bengio (2018). [Speaker Recognition from Raw Waveform with SincNet](https://arxiv.org/abs/1808.00158). IEEE SLT.
- Livingstone & Russo (2018). [RAVDESS](https://doi.org/10.1371/journal.pone.0196391). PLoS ONE.
