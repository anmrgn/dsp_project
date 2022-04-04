from email.mime import audio
from simulate_sound import Sim
import numpy as np
import os.path as osp
from proj_cfg import proj_cfg
import util
from time_delay import time_delay

def main():
    mic_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/mic_cfg.json")
    speaker_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/speaker_cfg.json")
    physics_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/physics_cfg.json")
    s = Sim(mic_cfg_fname, speaker_cfg_fname, physics_cfg_fname)

    rval = s.run({0 : np.array([1, 2, 3])})

    fS = s.get_sample_frequency()

    util.plot_audio_dat(rval, fS)

if __name__ == "__main__":
    main()

