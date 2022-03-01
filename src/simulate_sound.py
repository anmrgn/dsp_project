from mic_array import MicArray
import numpy as np
import json
import matplotlib.pyplot as plt
from speaker_array import SpeakerArray

class Sim:
    def __init__(self, mic_cfg_fname: str, physics_cfg_fname: str) -> None:
        """
        Setup simulator for sound localization
        """
        self.mic_array = MicArray(mic_cfg_fname)
        
        self.physics_cfg_fname = physics_cfg_fname
        self._load_physics_cfg()

    def sim_time_delay(self, speakers: SpeakerArray]) -> list[float]:
        """
        Takes a 2-d numpy array, where the first index is 
        """
        time_delays = []
        for mic_loc in self.mic_array.pos:
            time_delays.append(float(np.linalg.norm(mic_loc - src_loc) / self.speed_of_sound))
        
        return time_delays


    def _load_physics_cfg(self) -> None:

        with open(self.physics_cfg_fname) as f:
            data = json.load(f)

        self.speed_of_sound = data["speed_of_sound"]

    def show_mic_locs(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        ax.scatter(self.mic_array.pos[:, 0], self.mic_array.pos[:, 1], self.mic_array.pos[:, 2])

        plt.show()