import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose
import pandas as pd
import os.path as osp
from proj_cfg import proj_cfg
import pickle

use_dummy_data = False

class AngleDataset(Dataset):
    def __init__(self, X: torch.tensor, y: torch.tensor, mean: torch.tensor = None, std: torch.tensor = None) -> None:
        super().__init__()

        self.X = X
        self.y = y

        self.mean = mean
        self.std = std

        assert len(X) == len(y)

        self.N = len(X)
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        if self.mean is not None and self.std is not None:
            return (self.X[index] - self.mean) / self.std, self.y[index]
        else:
            return self.X[index], self.y[index]

class Model(nn.Module):
    def __init__(self, input_units, hidden_units, output_units, dropout_rate) -> None:
        super(Model, self).__init__()
        
        self.in_layer = nn.Linear(input_units, hidden_units)
        self.hidden_layer_1 = nn.Linear(hidden_units, hidden_units)
        self.output_layer = nn.Linear(hidden_units, output_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        
        x = self.in_layer(x)
        x = self.relu(x)
        x = self.hidden_layer_1(x)
        x = self.dropout(x)
        x = self.relu(x)
        y = self.output_layer(x)

        return y


def main():
    model_save_path = osp.join(proj_cfg["root_dir"], f"nn/{proj_cfg['angle_nn']}")

    plt.style.use('ggplot')

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Set the batch size
    batch_size = 64

    if use_dummy_data:
        data = pd.read_csv(osp.join(proj_cfg["root_dir"], f"dat/dummy.csv"))
    else:
        data = pd.read_csv(osp.join(proj_cfg["root_dir"], f"dat/{proj_cfg['angle_training_dat']}"))

    x_data = data[["td0", "td1", "td2", "td3", "td4", "td5"]]
    y_data = data[["theta", "phi"]]

    x_data = torch.from_numpy(x_data.to_numpy(dtype=np.float32))
    y_data = torch.from_numpy(y_data.to_numpy(dtype=np.float32))

    N = len(x_data)
    train_percent = 0.8

    split_idx = int(N * train_percent)
    
    x_train = x_data[:split_idx]
    y_train = y_data[:split_idx]

    x_val = x_data[split_idx:]
    y_val = y_data[split_idx:]

    # x_train_mean = x_train.mean(dim=0)
    # x_train_std = x_train.std(dim=0)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()

    time_delay_transform_file = osp.join(proj_cfg["root_dir"], f"nn/{proj_cfg['time_delay_transform']}")
    with open(time_delay_transform_file, "wb") as f:
        pickle.dump({"mean": x_train_mean, "std": x_train_std}, f)
    
    # Create data loaders
    train_dataloader = DataLoader(AngleDataset(x_train, y_train, x_train_mean, x_train_std), batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(AngleDataset(x_val, y_val, x_train_mean, x_train_std), batch_size=batch_size, num_workers=4)

    # Define in n out dims
    input_units = 6
    output_units = 2
    hidden_units = 100
    dropout_rate = 0.2

    # Define the model by subclassing nn.Module
    
    model = Model(input_units, hidden_units, output_units, dropout_rate)

    print(model)

    # Set up loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)

    # Set an exponential LR scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    #begin training


    # Aux function that runs validation and returns val loss/acc
    def validation(dataloader, model, loss_fn):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        num_batches = len(dataloader)
        model.eval() # Set the model on inference mode; dropout
        loss = 0
        with torch.no_grad(): # no_grad() skips the gradient computation; faster
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss += loss_fn(pred, y).item()

        # Avg loss accross
        loss /= num_batches
        return loss 


    epochs = 10
    size = len(train_dataloader.dataset)

    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        for batch, (x,y) in enumerate(train_dataloader):
            model.train()

            # Transfer data to device
            x, y = x.to(device), y.to(device)

            # Pass data through model
            y_pred = model(x)

            # Calculate loss 
            loss = loss_fn(y_pred, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 200 == 0:

                # Append to lists 
                train_loss.append(loss.item())
        
                # Do validation
                validation_loss = validation(val_dataloader, model, loss_fn)
                val_loss.append(validation_loss)
        
            # Log some info
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"Epoch = {epoch}, train loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

        # Step the LR scheduler
        scheduler.step()
    
    print(f"Train loss: {train_loss}")
    print(f"Val loss: {val_loss}")


    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    main()
