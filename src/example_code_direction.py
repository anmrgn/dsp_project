
from email.mime import audio
from simulate_sound import Sim
import numpy as np
import os.path as osp
from proj_cfg import proj_cfg
import util
from time_delay import time_delay
from get_direction import get_direction


def main():
    mic_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/mic_cfg.json")
    speaker_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/speaker_cfg.json")
    physics_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/physics_cfg.json")
    s = Sim(mic_cfg_fname, speaker_cfg_fname, physics_cfg_fname)

    rval = s.run({0 : np.array([1, 2, 3])})

    fS = s.get_sample_frequency()

    #util.plot_audio_dat(rval, fS)

    angle = [0,0,0,0]
    count = 0

    for (audio_signal) in rval.values():
        if count!=0:
            angle[count-1] = get_direction(audio_signal,last,fS)
        else:
            zerovect = audio_signal
        last = audio_signal
        count = count+1
        print(count)
    if count==4:
        print("here")
        angle[count-1] = get_direction(audio_signal,zerovect,fS)

    print(angle)

    util.plot_audio_dat(rval, fS)


if __name__ == "__main__":
    main()
