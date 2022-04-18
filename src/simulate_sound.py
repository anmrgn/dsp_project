from typing import Union
from exceptions import InvalidInput
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

    def get_sample_frequency(self) -> Union[float, None]:
        """
        Returns the sample frequency of the microphones and speakers (if all are the same), else None
        """
        mic_freq = self.mic_array.get_sample_freq()
        speaker_freq = self.speaker_array.get_sample_freq()
        if mic_freq == speaker_freq and mic_freq is not None and speaker_freq is not None:
            return mic_freq
        else:
            return None

    def run(self, audio_signals: dict[Union[int, str], Union[np.ndarray, list[float]]]) -> dict[Union[int, str], np.ndarray]:
        """
        Run simulation for audio signals corresponding to speaker names, return audio signals received by microphones

        Arguments:
        audio_signals -- dictionary from the name of a speaker to the sound it produces 

        Returns:
        captured_audio -- dictionary from the name of each mic to the audio signal it captures
        """
        mic_freq = self.mic_array.get_sample_freq()
        speaker_freq = self.speaker_array.get_sample_freq()
        assert mic_freq == speaker_freq and mic_freq is not None and speaker_freq is not None
        fS = mic_freq 

        audio_signals = {src: np.array(signal) for src, signal in audio_signals.items()}

        time_delays = {}

        transmitted_signal = {}

        for src in audio_signals:
            if src not in self.speaker_array:
                raise InvalidInput(f"Could not find speaker with name {src}. Either modify the speaker_cfg.json file or the input audio signals to this method.")

            speaker = self.speaker_array.name_to_speaker[src]

            time_delays[src] = {}
            transmitted_signal[src] = {}

            for mic in self.mic_array.mics:
                distance = np.linalg.norm(mic.pos - speaker.pos)
                time_delays[src][mic.name] = float(distance / self.speed_of_sound)

                decay = 1 / (distance ** self.decay_factor)
                transmitted_signal[src][mic.name] = decay * np.pad(audio_signals[src], ((int(fS * time_delays[src][mic.name]), 0),))
        
        signal_lengths = np.array([[len(transmitted_signal[src][mic.name]) for mic in self.mic_array.mics] for src in transmitted_signal])
            
        max_len = np.max(signal_lengths)

        captured_audio = {}

        for mic in self.mic_array.mics:
            captured_audio[mic.name] = np.zeros(max_len)

            for src in transmitted_signal:
                signal = transmitted_signal[src][mic.name]

                captured_audio[mic.name] += np.pad(signal, ((0, max_len - len(signal)),))


        return captured_audio


    def _load_physics_cfg(self) -> None:

        with open(self.physics_cfg_fname) as f:
            data = json.load(f)

        self.speed_of_sound = data["speed_of_sound"]
        self.decay_factor = data["decay_factor"]

    def show_mic_speaker_locs(self) -> None:
        pass