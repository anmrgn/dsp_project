from simulate_sound import Sim
import numpy as np
import os.path as osp
from proj_cfg import proj_cfg
from time_delay import time_delay
import tqdm

N = 100000 # number of datapoints to generate

def func(td0, td1, td2, td3, td4, td5):
    return td0 + td1*td2 - 5*td3*td3 - td4 + 8 * td5, 10 * td0**3 + td1

# def func(td0, td1, td2, td3, td4, td5):
#     return td1, td2 + td4

def main():
    data_fname = "dummy.csv"

    data_fpath = osp.join(proj_cfg["root_dir"], f"dat/{data_fname}")
    
    data = np.random.uniform(0, 1, (N, 6))

    with open(data_fpath, "w") as f:

        row = ",".join(["td0", "td1", "td2", "td3", "td4", "td5", "theta", "phi"]) + "\n"
        f.write(row)

        for tds in tqdm.tqdm(data):
            
            td0, td1, td2, td3, td4, td5 = tds

            theta, phi = func(td0, td1, td2, td3, td4, td5)

            row = ",".join([str(item) for item in [td0, td1, td2, td3, td4, td5, theta, phi]]) + "\n"
            
            f.write(row)
          


if __name__ == "__main__":
    main()


