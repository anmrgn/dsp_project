from typing import Union
import torch
from anglenn import Model
from proj_cfg import proj_cfg
import os.path as osp
import pickle
import numpy as np
from time_delay import time_delay
from scipy import interpolate
from simulate_sound import Sim
from visualization import visualize

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

def fix_angles(theta, phi):
    if theta < 0:
        phi += np.pi
        theta = np.abs(theta)
    
    while phi < 0:
        phi += 2 * np.pi
    
    while phi > 2 * np.pi:
        phi -= 2 * np.pi
    
    return theta, phi

def time_delays_to_angles(td0, td1, td2, td3, td4, td5):
    inp = (torch.tensor([td0, td1, td2, td3, td4, td5], dtype=torch.float32) - mean) / std
    res = model(inp)
    theta, phi = res
    return fix_angles(theta.item(), phi.item())


def augment(mic_dat: dict[Union[int, str], Union[list[float], np.ndarray]], resample: int):
    assert resample >= 1

    rval = {}

    for mic_name, dat in mic_dat.items():
        k   = np.linspace(0, len(dat) - 1, len(dat))
        tck = interpolate.splrep(k, dat , s=0)

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

    Visual representation of mic locations in x-y plane

                y
               |            
               |            
         1     |     3      
               |             
    ___________|____________ x
               |            
         0     |     2      
               |            
               |                       
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
    mic_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/mic_cfg.json")
    speaker_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/speaker_cfg.json")
    physics_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/physics_cfg.json")
    s = Sim(mic_cfg_fname, speaker_cfg_fname, physics_cfg_fname)
    fS = s.get_sample_frequency()

    set_r = 1
    set_theta = np.pi / 6
    set_phi = 3 * np.pi / 4

    set_x = set_r * np.sin(set_theta) * np.cos(set_phi)
    set_y = set_r * np.sin(set_theta) * np.sin(set_phi)
    set_z = set_r * np.cos(set_theta)

    s.speaker_array.set_speaker_locs({0: np.array([set_x, set_y, set_z])}) # expect theta = pi / 4, phi = pi / 4
    rval = s.run({0: np.array([1, 2, 3, 2, 1])})

    theta, phi = pred_angles(rval, fS)
    print(f"predicted theta = {theta}, phi = {phi}")
    print(f"expected theta = {set_theta}, phi = {set_phi}")
    visualize(theta, phi, actual_phi=set_phi, actual_theta=set_theta)


if __name__ == "__main__":
    main()
