from simulate_sound import Sim
import numpy as np
import os.path as osp
from proj_cfg import proj_cfg
from time_delay import time_delay
import tqdm

N = 10000 # number of datapoints to generate
rmin = 5  # minimum radius for speaker location
rmax = 50 # maximum radius for speaker location

def gen_random_locs(N: int, rmin: float, rmax: float):

    r = np.random.uniform(rmin, rmax, N)
    theta = np.random.uniform(0, np.pi, N)
    phi = np.random.uniform(0, 2 * np.pi, N)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    spherical = list(zip(r, theta, phi))
    cartesian = list(zip(x, y, z))

    return spherical, cartesian


def main():
    """
    Assumes 4 microphones, 1 speaker in the json files.

    Microphones must be named 0, 1, 2, 3 (int)
    Speaker must be named 0 (int)
    """
    data_fname = proj_cfg["angle_training_dat"]

    data_fpath = osp.join(proj_cfg["root_dir"], f"dat/{data_fname}")

    mic_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/mic_cfg.json")
    speaker_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/speaker_cfg.json")
    physics_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/physics_cfg.json")
    s = Sim(mic_cfg_fname, speaker_cfg_fname, physics_cfg_fname)
    fS = s.get_sample_frequency()

    
    spherical, cartesian = gen_random_locs(N, rmin, rmax)

    with open(data_fpath, "w") as f:

        row = ",".join(["td0", "td1", "td2", "td3", "td4", "td5", "theta", "phi"]) + "\n"
        f.write(row)

        for (r, theta, phi), (x, y, z) in tqdm.tqdm(zip(spherical, cartesian), total=N):
            s.speaker_array.set_speaker_locs({0: np.array([x, y, z])})

            rval = s.run({0 : np.array([1, 2, 3])})
            
            td0 = time_delay(rval[0], rval[1], fS)
            td1 = time_delay(rval[0], rval[2], fS)
            td2 = time_delay(rval[0], rval[3], fS)
            td3 = time_delay(rval[1], rval[2], fS)
            td4 = time_delay(rval[1], rval[3], fS)
            td5 = time_delay(rval[2], rval[3], fS)

            row = ",".join([str(item) for item in [td0, td1, td2, td3, td4, td5, theta, phi]]) + "\n"
            
            f.write(row)
          




if __name__ == "__main__":
    main()


