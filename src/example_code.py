from email.mime import audio
from simulate_sound import Sim
import numpy as np
import os.path as osp
from proj_cfg import proj_cfg
import json



s = Sim(osp.join(proj_cfg["root_dir"], "cfg/mic_cfg.json"), osp.join(proj_cfg["root_dir"], "cfg/speaker_cfg.json"), osp.join(proj_cfg["root_dir"], "cfg/physics_cfg.json"))

rval = s.run({0 : np.array([1,2,3])})