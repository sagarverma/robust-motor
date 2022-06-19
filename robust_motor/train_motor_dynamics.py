import numpy as np

import torch 
import torch.nn as nn 
import torch.optim as optim 

from tensorboardX import SummaryWriter

from robust_motor.utils.parser import get_args
from robust_motor.utils.helpers import get_paths, get_model, get_loss_function, log
from robust_motor.utils.metrics import smape, r2, rmse, mae
from robust_motor.datasets.motor_dynamics import get_loaders



args = get_args()

weight_dir, log_dir = get_paths(args)
writer = SummaryWriter(log_dir)

model = get_model(args)
train_loader, val_loader = get_loaders(args)

criterion = get_loss_function(args)
optimizer = optim.Adam(model.parameters())

best_smape = 1000000

for epoch in range(args.epochs):
    train_rmses = []
    train_maes = []
    train_r2s = []
    train_smapes = []
    train_losses = []

    model.train()

    for X, y in train_loader:
        X = X.cuda()
        y = y.cuda()

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        y = y.cpu().numpy()
        pred = pred.data.cpu().numpy()

        smape_err = smape(y, pred)
        r2_err = r2(y, pred)
        rmse_err = rmse(y, pred)
        mae_err = mae(y, pred)

        train_rmses.append(rmse_err)
        train_maes.append(mae_err)
        train_r2s.append(r2_err)
        train_smapes.append(smape_err)
        train_losses.append(loss.item())

    log(writer, epoch, 'train', np.mean(train_losses),
        {'RMSE': np.mean(train_rmses), 'MAE': np.mean(train_maes),
        'R2': np.mean(train_r2s), 'SMAPE': np.mean(train_smapes)})

