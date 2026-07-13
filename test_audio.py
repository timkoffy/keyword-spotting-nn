import torch
import torchaudio
import matplotlib.pyplot as plt

from model import KeywordSpottingModelV1
from dataset import KWS12Dataset

waveform, sr = torchaudio.load("./data/kws12_test_right.wav")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=160,
    n_mels=40
)

mel_spec = mel_transform(waveform)

if mel_spec.shape[2] > 100:
    mel_spec = mel_spec[:, :, :100]

mel_spec = torch.unsqueeze(mel_spec, 0)
print(mel_spec.shape)
print(waveform.shape, sr)

model = KeywordSpottingModelV1().to(torch.device("cpu"))
with torch.inference_mode():
    model.load_state_dict(torch.load("kws_model.pth"))
    model.eval()

    probs = model(mel_spec)
    print(probs.shape)
    probs = torch.softmax(probs, 1)
    print(probs.shape)

    print(probs)

    labels = KWS12Dataset.LABELS + ["unknown", "silence"]

    value, idx = torch.max(probs, 1)
    print(value, idx)

    plt.plot(labels, probs.squeeze())
    plt.xticks(rotation=45)
    plt.vlines(labels[idx], 0, 1, color="r", linestyles="dashed")
    plt.show()
