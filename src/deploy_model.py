from typing import Union
import torch
from anglenn import Model
from proj_cfg import proj_cfg
import os.path as osp
import pickle
import numpy as np
from time_delay import time_delay
from scipy import interpolate

input_units = 6
output_units = 2
hidden_units = 100
dropout_rate = 0.2

model_save_path = osp.join(proj_cfg["root_dir"], f"nn/{proj_cfg['angle_nn']}")

model = Model(input_units, hidden_units, output_units, dropout_rate)
model.load_state_dict(torch.load(model_save_path))
model.eval()

time_delay_transform_file = osp.join(proj_cfg["root_dir"], f"nn/{proj_cfg['time_delay_transform']}")
with open(time_delay_transform_file, "rb") as f:
    data = pickle.load(f)

    mean = data["mean"]
    std = data["std"]

def time_delays_to_angles(td0, td1, td2, td3, td4, td5):
    inp = (torch.tensor([td0, td1, td2, td3, td4, td5], dtype=torch.float32) - mean) / std
    res = model(inp)
    theta, phi = res
    return theta.item(), phi.item()


def augment(mic_dat: dict[Union[int, str], Union[list[float], np.ndarray]], resample: int):
    assert resample >= 1

    rval = {}

    for mic_name, dat in mic_dat.items():
        k     = np.linspace(0, len(dat) - 1, len(dat))
        tck = interpolate.splrep(k, dat)

        k_new = np.linspace(0, len(dat) - 1, resample * (len(dat) - 1) + 1)
        augmented_mic_dat = interpolate.splev(k_new, tck, der=0)

        rval[mic_name] = augmented_mic_dat
    
    return rval




def pred_angles(mic_dat: dict[Union[int, str], Union[list[float], np.ndarray]], fS: int, resample: int = 5):
    """
    Assumes microphones are oriented at locations
    Mic 0: [-0.0289, -0.0289, 0]
    Mic 1: [-0.0289,  0.0289, 0]
    Mic 2: [ 0.0289, -0.0289, 0]
    Mic 3: [ 0.0289,  0.0289, 0]

    Microphone names must be integers (0, 1, 2, 3)

    Resample specifies how to augment the mic data, i.e. for each one data point becomes resample data points after augmentation

    Returns theta, phi
    """
    mic_dat = augment(mic_dat, resample)
    fS *= resample

    td0 = time_delay(mic_dat[0], mic_dat[1], fS)
    td1 = time_delay(mic_dat[0], mic_dat[2], fS)
    td2 = time_delay(mic_dat[0], mic_dat[3], fS)
    td3 = time_delay(mic_dat[1], mic_dat[2], fS)
    td4 = time_delay(mic_dat[1], mic_dat[3], fS)
    td5 = time_delay(mic_dat[2], mic_dat[3], fS)

    return time_delays_to_angles(td0, td1, td2, td3, td4, td5)

def main():
    print(pred_angles({idx: np.array([1,2,3,5,6,4,4,3]) for idx in range(4)}, 44000))


if __name__ == "__main__":
    main()
