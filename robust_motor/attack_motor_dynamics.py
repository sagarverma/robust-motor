import tqdm
import argparse as ag
import tqdm

import numpy as np
import torch 
import torch.nn as nn 

from advertorch.attacks import GradientSignAttack, L2PGDAttack

from robust_motor.utils.metrics import smape, r2, rmse, mae
from robust_motor.utils.helpers import get_model
from robust_motor.datasets.motor_dynamics import get_loaders


parser = ag.ArgumentParser(description='Attack params')

parser.add_argument('--weight_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--attack', action='store_true')

args = parser.parse_args()
    
dataset_name = args.weight_path.split('/')[-3]
model_name = args.weight_path.split('/')[-2]

args.model = model_name
args.dataset = dataset_name 

model = get_model(args)
weight = torch.load(args.weight_path, map_location=torch.device(args.gpu))
model.load_state_dict(weight)
# model.eval()

_, val_loader = get_loaders(args)


fgm_attack = GradientSignAttack(model, loss_fn=nn.MSELoss(), eps=0.1)
pgd_attack = L2PGDAttack(model, loss_fn=nn.MSELoss(), eps=0.1, eps_iter=0.1/3, nb_iter=20)

fgm_smape = []
fgm_r2 = []
fgm_rmse = []
fgm_mae = []

pgd_smape = []
pgd_r2 = []
pgd_rmse = []
pgd_mae = []

clean_smape = []
clean_r2 = []
clean_rmse = []
clean_mae = []

i = 0
for X, y in tqdm.tqdm(val_loader):
    X = X.to(args.gpu)
    y = y.to(args.gpu)
    
    fgm_adv = fgm_attack.perturb(X, y)
    pgd_adv = pgd_attack.perturb(X, y)

    clean_pred = model(X)
    fgm_pred = model(fgm_adv)
    pgd_pred = model(pgd_adv)

    y = y.cpu().numpy()
    clean_pred = clean_pred.data.cpu().numpy()
    fgm_pred = fgm_pred.data.cpu().numpy()
    pgd_pred = pgd_pred.data.cpu().numpy()

    clean_smape.append(smape(y, clean_pred))
    clean_r2.append(r2(y, clean_pred))
    clean_rmse.append(rmse(y, clean_pred))
    clean_mae.append(mae(y, clean_pred))

    fgm_smape.append(smape(y, fgm_pred))
    fgm_r2.append(r2(y, fgm_pred))
    fgm_rmse.append(rmse(y, fgm_pred))
    fgm_mae.append(mae(y, fgm_pred))
    
    pgd_smape.append(smape(y, pgd_pred))
    pgd_r2.append(r2(y, pgd_pred))
    pgd_rmse.append(rmse(y, pgd_pred))
    pgd_mae.append(mae(y, pgd_pred))
    
    i += 1
    if i == 100:
        break

fgm_mae = np.mean(fgm_mae)
fgm_r2 = np.mean(fgm_r2)
fgm_rmse = np.mean(fgm_rmse)
fgm_smape = np.mean(fgm_smape)

pgd_mae = np.mean(pgd_mae)
pgd_r2 = np.mean(pgd_r2)
pgd_rmse = np.mean(pgd_rmse)
pgd_smape = np.mean(pgd_smape)

clean_mae = np.mean(clean_mae)
clean_r2 = np.mean(clean_r2)
clean_rmse = np.mean(clean_rmse)
clean_smape = np.mean(clean_smape)

print (f'Clean Acc: R2 {clean_r2}, SMAPE {clean_smape}, RMSE {clean_rmse}, MAE {clean_mae}')
print (f'PGD: R2 {pgd_r2}, SMAPE {pgd_smape}, RMSE {pgd_rmse}, MAE {pgd_mae}')
print (f'Fast Gradient Method: R2 {fgm_r2}, SMAPE {fgm_smape}, RMSE {fgm_rmse}, MAE {fgm_mae}')


fout = open(args.weight_path.replace('.pt', '_attack.txt'), 'w')
fout.write(f'Clean Acc: R2 {clean_r2}, SMAPE {clean_smape}, RMSE {clean_rmse}, MAE {clean_mae}\n')
fout.write(f'PGD: R2 {pgd_r2}, SMAPE {pgd_smape}, RMSE {pgd_rmse}, MAE {pgd_mae}\n')
fout.write(f'Fast Gradient Method: R2 {fgm_r2}, SMAPE {fgm_smape}, RMSE {fgm_rmse}, MAE {fgm_mae}\n')
fout.close()
