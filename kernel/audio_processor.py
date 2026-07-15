import torch
import torchaudio
import numpy as np

from model import KeywordSpottingModelV1
from config import (
    TARGET_SAMPLE_RATE, BUFFER_SIZE, MODEL_PATH, DEVICE, 
    LABELS, CONFIDENCE_THRESHOLD, PROCESS_EVERY_N
)


class AudioProcessor:
    def __init__(self):
        self.device = DEVICE
        self.model = self._load_model()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE, n_fft=512, hop_length=160, n_mels=40
        )
        
        self.buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.chunk_count = 0
        self.last_detected = None
        self.cooldown = 0

    def _load_model(self):
        model = KeywordSpottingModelV1(num_classes=12).to(self.device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        model.eval()
        return model

    def process_chunk(self, chunk_bytes: bytes):
        chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        self.buffer[:-len(chunk_np)] = self.buffer[len(chunk_np):]
        self.buffer[-len(chunk_np):] = chunk_np
        self.chunk_count += 1

        if self.chunk_count % PROCESS_EVERY_N != 0:
            return None

        waveform = torch.tensor(self.buffer, dtype=torch.float32).unsqueeze(0)
        mel_spec = self.mel_transform(waveform)
        
        if mel_spec.shape[2] > 100:
            mel_spec = mel_spec[:, :, :100]
        elif mel_spec.shape[2] < 100:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 100 - mel_spec.shape[2]))
            
        mel_spec = mel_spec.unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            output = self.model(mel_spec)
            probs = torch.softmax(output, 1)
            confidence, predicted = torch.max(probs, 1)
            
            label_idx = predicted.item()
            label = LABELS[label_idx]
            conf_val = confidence.item()
            
            if label_idx < 10 and conf_val > CONFIDENCE_THRESHOLD:
                if label != self.last_detected or self.cooldown <= 0:
                    self.last_detected = label
                    self.cooldown = 3
                    return label
                    
        if self.cooldown > 0:
            self.cooldown -= 1
            
        return None
