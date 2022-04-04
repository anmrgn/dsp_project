import json
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

def plot_audio_dat(audio_dat: dict[Union[int, str], np.ndarray], fS: float) -> None:
    """
    Plot audio data in the form of a dictionary from speaker/mic name to audio data signal
    """
    fig, axs = plt.subplots(len(audio_dat))
    fig.suptitle("Audio data")
    
    for ax, (name, audio_signal) in zip(axs, audio_dat.items()):
        t = np.array(range(len(audio_signal))) / fS
        ax.set_title(f"Audio source from speaker/microphone {name}")
        ax.plot(t, audio_signal)
        ax.set(xlabel="Time (s)", ylabel="Signal")

    plt.tight_layout()
    plt.show()
    


def pprint_audio_dat(audio_dat: dict[Union[int, str], np.ndarray]) -> None:
    """
    Pretty print audio data in the form of a dictionary from speaker/mic name to audio data signal
    """
    json_audio_dat = {name: list(audio_signal) for name, audio_signal in audio_dat.items()}
    print(json.dumps(json_audio_dat, indent=4))