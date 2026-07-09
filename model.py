import torch.nn as nn

class KeywordSpottingModelV0(nn.Module):
    """
    2D CNN for Keyword Spotting.
    Treats the Mel-spectrogram [1, 40, 100] as a single-channel image.
    """
    def __init__(self, num_classes=12):
        super().__init__()

        self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # output: [B, 32, 20, 50]

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # output: [B, 64, 10, 25]
                
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1) # output: [B, 128, 1, 1]
                )
    
        self.classifier = nn.Sequential(
                nn.Flatten(), # output: [B, 128]
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(64, num_classes) # output: [B, 12]
                )

        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    import torch
    model = KeywordSpottingModelV0()
    dummy_input = torch.randn(4, 1, 40, 100)
    output = model(dummy_input)
    print(dummy_input.shape)
    print(output.shape)
    assert output.shape == (4, 12)

