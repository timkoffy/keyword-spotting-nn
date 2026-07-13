import matplotlib.pyplot as plt
import torchaudio
import torch
import numpy as np
import os
import sys
from log import TrainingStats

def plot_mel_spec(mel_spec, label=None):
    """
    Plot a single Mel-spectrogram with an optional label.

    Args:
        mel_spec (torch.Tensor): The Mel-spectrogram tensor of shape [H, W] or [1, H, W].
        label (str, optional): A string label to display in the plot title.
    """
    if mel_spec is None:
        print("Mel spec has not been provided")
        return

    if mel_spec.dim() == 3:
        mel_spec = mel_spec[0] 
   
    power_to_db = torchaudio.transforms.AmplitudeToDB("power", 80.0)
    
    plt.imshow(power_to_db(mel_spec), cmap="plasma", origin="lower", interpolation="nearest")
    plt.title(f"Spectrogram of '{label}'" if label else "Spectrogram")

    plt.show()


def plot_mel_specs_multiply(mel_specs, labels=None):
    """
    Plot a grid of Mel-spectrograms for batch visualization and comparison.

    Args:
        mel_specs (torch.Tensor): A batch of Mel-spectrograms of shape [B, H, W] or [B, 1, H, W].
        labels (list[str], optional): A list of string labels corresponding to each spectrogram in the batch.
    """
    n_batches = mel_specs.shape[0]
    
    if mel_specs.dim() == 4:
        mel_specs = torch.squeeze(mel_specs, dim=1)
    
    power_to_db = torchaudio.transforms.AmplitudeToDB("power", 80.0)

    n_cols = max(1, n_batches // 4)
    n_rows = (n_batches + n_cols - 1) // n_cols 

    f, axs = plt.subplots(n_rows, n_cols)

    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs[np.newaxis, :]
    elif n_cols == 1:
        axs = axs[:, np.newaxis]

    f.set_figheight(3 * n_rows)
    f.set_figwidth(4 * n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            
            if idx >= n_batches:
                axs[i][j].axis('off')
                continue

            mel_spec = mel_specs[idx]

            axs[i][j].imshow(power_to_db(mel_spec).cpu().numpy(), cmap="plasma", origin="lower", interpolation="nearest")
            
            if labels is not None:
                axs[i][j].set_title(labels[idx])

    plt.tight_layout()
    plt.show()


def plot_loss_stats(path="training_stats.pkl"):
    stats = TrainingStats.load(path)

    history = stats.history

    plt.plot(history["epochs"], history["train_loss"], label="Train Loss")
    plt.plot(history["epochs"], history["val_loss"], label="Validate Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_acc_stats(path="training_stats.pkl"):
    stats = TrainingStats.load(path)

    history = stats.history

    plt.plot(history["epochs"], history["train_acc"], label="Train Accuracy")
    plt.plot(history["epochs"], history["val_acc"], label="Validate Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy, %")
    plt.ylim(0, 100)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_path = "./data/mel_spec_test_data/test_C_octaves.wav"
    if not (os.path.exists(file_path) and file_path.lower().endswith('.wav')):
        print(f"Error: File not found or invalid format at '{file_path}'")
        sys.exit(1)

    waveform, sr = torchaudio.load(file_path)
    
    n_fft = 1024
    win_length = None
    hop_length = sr // 100
    n_mels = 256

    mel_spec_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=sr // 2,
        win_length=win_length,
        hop_length=hop_length,
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )

    mel_spec = mel_spec_fn(waveform)

    plot_mel_spec(mel_spec, label="test_C_octaves")

    plot_acc_stats("training_stats.pkl")
