from email.mime import audio
from simulate_sound import Sim
import numpy as np
import os.path as osp
from proj_cfg import proj_cfg




s = Sim(osp.join(proj_cfg["root_dir"], "cfg/mic_cfg.json"), osp.join(proj_cfg["root_dir"], "cfg/physics_cfg.json"))

audio_src_loc = np.array([1, 2, 3])

print(s.sim_time_delay(audio_src_loc))

s.show_mic_locs()