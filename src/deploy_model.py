import torch
from anglenn import Model
from proj_cfg import proj_cfg
import os.path as osp
import pickle

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

def pred_angles(td0, td1, td2, td3, td4, td5):
    inp = (torch.tensor([td0, td1, td2, td3, td4, td5]) - mean) / std
    res = model(inp)
    theta, phi = res
    return theta.item(), phi.item()
