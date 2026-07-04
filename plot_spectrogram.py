import matplotlib.pyplot as plt
import torchaudio
import torch

# todo: normalise freqs and time on plot 
def plot_mel_spec(mel_spec, label=None) :
    """
    Plotting mel spectrogram + waveform with labeling (input: tensor [H, W] or [1, H, W])
    """
    if mel_spec is None:
        print("Mel spec has not given")
        return

    if mel_spec.dim() == 3:
        mel_spec = mel_spec[0] 
   
    power_to_db = torchaudio.transforms.AmplitudeToDB("power", 80.0)
    
    plt.imshow(power_to_db(mel_spec), cmap="plasma", origin="lower", interpolation="nearest")
    plt.title(f"Spectrogram of '{label}'" if label else "Spectrogram")

    plt.show()

def plot_mel_specs_multiply(mel_specs, labels=None):
    """
    Plotting mel spectrograms + waveforms with labeling for comparing (input: tensor [B, H, W] or [B, 1, H, W])
    """
    n_batches = mel_specs.shape[0]
    
    if mel_specs.dim() == 4:
        mel_specs = torch.squeeze(mel_specs, dim=1)
    
    power_to_db = torchaudio.transforms.AmplitudeToDB("power", 80.0)

    n_cols = n_batches // 4
    n_rows = n_batches // n_cols

    _, axs = plt.subplots(n_rows, n_cols)

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

    plt.tight_layout(pad=0.5)
    plt.show()


if __name__ == "__main__":
    waveform, sr = torchaudio.load('./test_data/test_C_octaves.wav')
    
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
