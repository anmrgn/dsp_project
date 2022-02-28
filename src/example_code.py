from email.mime import audio
from simulate_sound import Sim
import numpy as np


s = Sim("C:\homework\dsp_project\cfg\mic_cfg.json", "C:\homework\dsp_project\cfg\physics_cfg.json")

audio_src_loc = np.array([1, 2, 3])

print(s.sim_time_delay(audio_src_loc))

s.show_mic_locs()