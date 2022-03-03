from typing import Union
from mic_array import MicArray
from speaker_array import SpeakerArray
import numpy as np
import json
import matplotlib.pyplot as plt

class Sim:
    def __init__(self, mic_cfg_fname: str, speaker_cfg_fname: str, physics_cfg_fname: str) -> None:
        """
        Setup simulator for sound localization
        """
        self.mic_array = MicArray(mic_cfg_fname)
        self.speaker_array = SpeakerArray(speaker_cfg_fname)
        
        self.physics_cfg_fname = physics_cfg_fname
        self._load_physics_cfg()

    def run(self, audio_signals: dict[Union[int, str], np.ndarray]) -> dict[Union[int, str], np.ndarray]:
        """
        Runs simulation for audio signals corresponding to speaker names, returns audio signals received by microphones

        Arguments:
        audio_signals -- dictionary from the name of a speaker to the sound it produces 
        """
        mic_freq = self.mic_array.get_sample_freq()
        speaker_freq = self.speaker_array.get_sample_freq()
        assert mic_freq == speaker_freq and mic_freq is not None and speaker_freq is not None

        time_delays = {}

        recv_signals = {}

        
        for src in audio_signals:
            assert src in self.speaker_array

            speaker = self.speaker_array.name_to_speaker[src]

            time_delays[src] = {}
            recv_signals[src] = {}

            for mic in self.mic_array.mics:
                time_delays[src][mic.name] = float(np.linalg.norm(mic.pos - speaker.pos) / self.speed_of_sound)
                recv_signals[src] = np.pad()
            
        
        captured_audio = 

        

        # for mic_loc in self.mic_array.pos:
        #     time_delays.append(float(np.linalg.norm(mic_loc - src_loc) / self.speed_of_sound))
        
        return None


    def _load_physics_cfg(self) -> None:

        with open(self.physics_cfg_fname) as f:
            data = json.load(f)

        self.speed_of_sound = data["speed_of_sound"]

    def show_mic_locs(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        ax.scatter(self.mic_array.pos[:, 0], self.mic_array.pos[:, 1], self.mic_array.pos[:, 2])

        plt.show()