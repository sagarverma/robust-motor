import tqdm
import numpy as np

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score 

from tensorboardX import SummaryWriter

from robust_motor.utils.parser import get_args
from robust_motor.utils.helpers import get_paths, get_model, log
from robust_motor.utils.metrics import smape, r2, rmse, mae
from robust_motor.datasets.temperature import get_loaders


args = get_args()

weight_dir, log_dir = get_paths(args)
writer = SummaryWriter(log_dir)

model = get_model(args)
train_loader, val_loader = get_loaders(args)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)


best_smape = 1000000

for epoch in range(args.epochs):
    train_rmses = []
    train_maes = []
    train_r2s = []
    train_smapes = []
    train_losses = []

    model.train()

    for X, y in tqdm.tqdm(train_loader):
        X = X.cuda(args.gpu)
        y = y.cuda(args.gpu)

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

    print ("\n\n#############################################\n\n")
    log(writer, epoch, 'train', np.mean(train_losses),
        {'RMSE': np.mean(train_rmses), 'MAE': np.mean(train_maes),
        'R2': np.mean(train_r2s), 'SMAPE': np.mean(train_smapes)})

    model.eval()
    with torch.no_grad():
        val_rmses = []
        val_maes = []
        val_r2s = []
        val_smapes = []
        val_losses = []
        
        for X, y in tqdm.tqdm(val_loader):
            X = X.cuda(args.gpu)
            y = y.cuda(args.gpu)

            pred = model(X)
            loss = criterion(pred, y)
            y = y.cpu().numpy()
            pred = pred.data.cpu().numpy()

            smape_err = smape(y, pred)
            r2_err = r2(y, pred)
            rmse_err = rmse(y, pred)
            mae_err = mae(y, pred)

            val_rmses.append(rmse_err)
            val_maes.append(mae_err)
            val_r2s.append(r2_err)
            val_smapes.append(smape_err)
            val_losses.append(loss.item())

        scheduler.step(np.mean(val_smapes))
        print ("\n")
        log(writer, epoch, 'val', np.mean(val_losses),
            {'RMSE': np.mean(val_rmses), 'MAE': np.mean(val_maes),
            'R2': np.mean(val_r2s), 'SMAPE': np.mean(val_smapes)})

    if np.mean(val_smapes) < best_smape:
        best_smape = np.mean(val_smapes)
        torch.save( model.state_dict(), f"{weight_dir}/checkpoint-{str(epoch).zfill(3)}.pt")