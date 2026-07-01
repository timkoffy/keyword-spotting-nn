import torch.nn as nn

class KeywordSpottingModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        # [batch, 1, 40, 100]
        self.features = nn.Sequential()
