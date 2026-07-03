import matplotlib.pyplot as plt

# todo: normalise freqs and time on plot 
# todo: allow to plot entire batch of melspecs for comparing
class PlotMelSpec():
    def __init__(self,
                 mel_spec=None, 
                 label=None) :
        """
        Plotting mel spectrogram (tensor [H, W] or [1, H, W]) with labeling and coloring
        """

        if mel_spec is None:
            print("Mel spec tensor has not given")
            return

        mel_spec = mel_spec.squeeze()
        plt.imshow(mel_spec, cmap="plasma", interpolation="bilinear")

        if label: 
            plt.title(f"Mel-spectrogram of '{label}'")

        plt.colorbar()

        plt.xlabel("time")
        plt.ylabel("freq")

        plt.show()


if __name__ == "__main__":
    PlotMelSpec(label="bruh")
