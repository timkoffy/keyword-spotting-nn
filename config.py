import torch

TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION = 0.064
CHUNK_SIZE = int(TARGET_SAMPLE_RATE * CHUNK_DURATION)
BUFFER_SIZE = TARGET_SAMPLE_RATE
PROCESS_EVERY_N = 8

CONFIDENCE_THRESHOLD = 0.75
MODEL_PATH = "models/kws_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "yes", "no", "up", "down", "left", 
    "right", "on", "off", "stop", "go", 
    "unknown", "silence"
]
