import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from plot_mel_specs import PlotMelSpec as pltms


class KWS12Dataset(Dataset):
    """

    """

    LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
    UNKNOWN_IDX = 10
    SILENCE_IDX = 11

    def __init__(
            self, 
            root="./data", 
            subset="training", 
            augment=False,
            unknown_ratio=0.1,
            silence_ratio=0.1,
            seed=42
    ):
        self.root = root
        self.subset = subset
        self.augment = augment
        self.seed = seed
        
        random.seed(seed)
        torch.manual_seed(seed)

        self.base_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            url="speech_commands_v0.02",
            download=True,
            subset=subset
        )   
    
        keyword_samples = defaultdict(list)
        unknown_samples = []
        background_noise_files = []

        noise_dir = os.path.join(root, "SpeechCommands", "speech_commands_v0.02", "_background_noise_")
        if os.path.exists(noise_dir):
            background_noise_files = glob.glob(os.path.join(noise_dir, "*wav"))

        
        walker = getattr(self.base_dataset, '_walker', [])
        
        for path in walker:
            label = os.path.basename(os.path.dirname(path))
            if label in self.LABELS:
                keyword_samples[label].append(path)
            elif label == "_background_noise_":
                background_noise_files.append(path)
            else:
                unknown_samples.append(path)

        class_sizes = {label: len(paths) for label, paths in keyword_samples.items()}
        base_size = max(class_sizes.values())
        unknown_size = int(base_size * unknown_ratio)
        self.silence_size = int(base_size * silence_ratio)

        sampled_unknown = random.sample(unknown_samples, unknown_size)

        self.samples = []
        for label, paths in keyword_samples.items():
            for p in paths:
                self.samples.append((p, self.LABEL_TO_IDX[label]))
        for p in sampled_unknown:
            self.samples.append((p, self.UNKNOWN_IDX))

        self.background_noise_files = background_noise_files

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=40
        )
 
        print(f"=== Stats for KWS12Dataset ({subset}) ===")
        print(f"Base size (max keyword class): {base_size}")
        for label in self.LABELS:
            print(f"    {label:5s}: {class_sizes.get(label, 0)}")
        print(f"Unknown: {len(unknown_samples)} -> sampled {unknown_size}")
        print(f"Silence (virtual): {self.silence_size}")
        print(f"Total dataset length: {len(self)}")
        print("=========================================")

    def __len__(self):
        return len(self.samples) + self.silence_size

    def __getitem__(self, idx):
        if idx < len(self.samples):
            path, label = self.samples[idx]
            waveform, _ = torchaudio.load(path)
        else: 
            if self.background_noise_files:
                noise_path = random.choice(self.background_noise_files)
                waveform, _ = torchaudio.load(noise_path)
                
                start = random.randint(0, waveform.shape[1] - 16000)
                waveform = waveform[:, start:start + 16000]
            else: 
                waveform = torch.zeros(1, 16000)
            label = self.SILENCE_IDX

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] > 16000:
            waveform = waveform[:, :16000]
        elif waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))


        if self.augment:
            shift = random.randint(-1600, 1600) # ~100 ms
            if shift > 0:
                waveform = torch.nn.functional.pad(waveform, (shift, 0))
            elif shift < 0: 
                waveform = torch.nn.functional.pad(waveform, (0, -shift))

            vol = random.uniform(0.8, 1.2)
            waveform = waveform * vol


        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)

        if mel_spec.shape[2] > 100:
            mel_spec = mel_spec[:, :, :100]
        elif mel_spec.shape[2] < 100:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 100 - mel_spec.shape[2]))

        
        return mel_spec, label



if __name__ == "__main__":
    dataset = KWS12Dataset()

    loader = DataLoader(
        dataset,
        batch_size=64, 
        shuffle=True,
        num_workers=4,
    )

    for mels, labels in loader:
        assert mels.shape == (64, 1, 40, 100), f"Unexpected mel shape: {mels.shape}"
        assert labels.shape == (64,), f"Unexpected labels shape: {labels.shape}"
        print(f"Batch processed. Mel: {mels.shape}, Labels: {labels.shape}")
        
        pltms(mels[0], labels[0])

        break
