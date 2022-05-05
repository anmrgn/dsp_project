from typing import Union
import torch
from anglenn import Model
from proj_cfg import proj_cfg
import os.path as osp
import numpy as np
import json

from deploy_model import *
from raw_audio_data import Get_Raw_Audio

input_units = 6
output_units = 2
hidden_units = 100
dropout_rate = 0.2


def main():

    fS = 16000
    
    data_sample_dict = Get_Raw_Audio()

    for i in range(200):

        data_sample_dict_truncated = {0:data_sample_dict[0][(i+1)*0:(i+1)*199],
            1:data_sample_dict[1][(i+1)*0:(i+1)*199],
            2:data_sample_dict[2][(i+1)*0:(i+1)*199],
            3:data_sample_dict[3][(i+1)*0:(i+1)*199]}
        print("new dict size = ")
        print(np.size(data_sample_dict_truncated[0]))
        theta, phi = pred_angles(data_sample_dict_truncated, fS)
        print(f"predicted theta = {theta}, phi = {phi}")
        print(f"expected theta = {torch.pi / 4}, phi = {torch.pi / 2}")


if __name__ == "__main__":
    main()
