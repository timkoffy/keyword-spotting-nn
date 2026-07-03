import matplotlib.pyplot as plt
import torch


class PlotMelSpec():
    def __init__(self,
                 mel_spec=None, 
                 label=None) :
        """
        Plotting mel spectrogram (tensor [H, W] or [1, H, W]) with labeling and coloring
        """
    
        mel_spec=torch.rand(1, 40, 100).squeeze().numpy()
        plt.imshow(mel_spec, interpolation="bilinear")

        if label: 
            plt.title(f"Mel-spectrogram of '{label}'")


        plt.show()


if __name__ == "__main__":
    PlotMelSpec(label="bruh")
