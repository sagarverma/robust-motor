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
from robust_motor.datasets.broken_bars import get_loaders


args = get_args()

weight_dir, log_dir = get_paths(args)
writer = SummaryWriter(log_dir)

model = get_model(args)
print (model)
train_loader, val_loader = get_loaders(args)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)

best_f1 = 0

for epoch in range(args.epochs):
    train_accs = []
    train_f1s = []
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
        pred = pred.argmax(axis=1)

        f1 = f1_score(y, pred, average='macro')
        acc = accuracy_score(y, pred)
        
        train_f1s.append(f1)
        train_accs.append(acc)
        train_losses.append(loss.item())

    print ("\n\n#############################################\n\n")
    log(writer, epoch, 'train', np.mean(train_losses),
        {'ACC': np.mean(train_accs), 'F1': np.mean(train_f1s)})

    model.eval()
    with torch.no_grad():
        val_accs = []
        val_f1s = []
        val_losses = []
        
        for X, y in tqdm.tqdm(val_loader):
            X = X.cuda(args.gpu)
            y = y.cuda(args.gpu)

            pred = model(X)
            loss = criterion(pred, y)
            y = y.cpu().numpy()
            pred = pred.data.cpu().numpy()
            pred = pred.argmax(axis=1)

            f1 = f1_score(y, pred, average='macro')
            acc = accuracy_score(y, pred)

            val_f1s.append(f1)
            val_accs.append(acc)
            val_losses.append(loss.item())

        scheduler.step(np.mean(val_f1s))
        print ("\n")
        log(writer, epoch, 'val', np.mean(val_losses),
            {'ACC': np.mean(val_accs), 'F1': np.mean(val_f1s)})

    if np.mean(val_f1s) > best_f1:
        best_f1 = np.mean(val_f1s)
        torch.save( model.state_dict(), f"{weight_dir}/checkpoint-{str(epoch).zfill(3)}.pt")