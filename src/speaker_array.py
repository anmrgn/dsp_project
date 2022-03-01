import json
from ctypes import Union
from typing import Optional
from exceptions import *
import numpy as np


class Speaker:
    """
    Represents a single speaker in the speaker array
    """
    required_params = ["pos"]

    def __init__(self, json_speaker_obj: dict, default_name: Optional[Union[int, str]] = None) -> None:
        """
        Takes a json-represented speaker object and generates a Speaker object

        Arguments:
        json_speaker_obj -- a dictionary that comes from parsing a json object representing a single speaker
        """

        for param in Speaker.required_params:
            if param not in json_speaker_obj:
                raise ConfigError(f"Speaker config json missing required parameter {param}, attempted to parse json object:\n {json.dumps(json_speaker_obj, indent=4)}")
        
        self.name : str        = json_speaker_obj["name"] if "name" in json_speaker_obj else default_name
        self.pos  : np.ndarray = np.array(json_speaker_obj["pos"])

class SpeakerArray:
    """
    Form a speaker array object from json specification.

    Attributes:
    nChannels -- the number of speakers/channels in the array (assumed each speaker is 1 channel)
    speakers  -- an array of speaker objects corresponding to speakers in the array
    """

    def __init__(self, cfg_fname: str) -> None:
        """
        Takes and loads json speaker configuration file that specifies speaker array.

        Arguments:
        cfg_fname -- a config filename specifying the speaker array. See speaker_cfg.json for an example.
        """
        self.cfg_fname = cfg_fname
        self._load_cfg()

        # None if validate_names has not been called, else True if names are valid, False if not.
        self.valid_names = None

    def _load_cfg(self) -> None:
        
        with open(self.cfg_fname) as f:
            data = json.load(f)
        
        self.nChannels = len(data) # i.e. number of microphones

        self.speakers: list[Speaker] = []

        for json_speaker_obj in data:
            self.speakers.append(Speaker(json_speaker_obj))
    
    def validate_names(self) -> bool:
        """
        Returns true if each mic has a valid name, false if not. Checks to ensure names are unique. Does not check for this, but all names must be a string or int (all names must have the same type).

        Call this method if you need each microphone to have a valid name. Otherwise, mics can be accessed by index.
        """

        names = set()

        for speaker in self.speakers:
            if speaker.name is None or speaker.name in names:
                self.valid_names = False
                return self.valid_names
            names.add(speaker.name)
        
        self.valid_names = True
        return self.valid_names
    
    def gen_names(self) -> None:
        """
        Generates names for each microphone (each mic has an integer name, 0, 1, etc.)
        """
        
        for idx, speaker in enumerate(self.speakers):
            speaker.name = idx
        