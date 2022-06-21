import tqdm
import argparse as ag
import tqdm

import numpy as np
import torch 
import torch.nn as nn 

from sklearn.metrics import f1_score, accuracy_score 

import foolbox as fb

from robust_motor.utils.helpers import get_model
from robust_motor.datasets.broken_bars import get_loaders


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
model.eval()

fmodel = fb.PyTorchModel(model, bounds=(0, 1))

_, val_loader = get_loaders(args)


fgm_attack = fb.attacks.L2FastGradientAttack()
df_attack = fb.attacks.L2DeepFoolAttack()

fgm_accs = []
df_accs = []
clean_accs = []

i = 0
for X, y in tqdm.tqdm(val_loader):
    X = X.to(args.gpu)
    y = y.to(args.gpu)
    
    pred = model(X).data.cpu().numpy()
    pred = pred.argmax(axis=1)
    clean_accs.append(accuracy_score(y.cpu().numpy(), pred))
    
    raw, clipped, is_adv_fgm = fgm_attack(fmodel, X, y, epsilons=0.1)
    raw, clipped, is_adv_df = df_attack(fmodel, X, y, epsilons=0.1)

    fgm_accs.append((1 - is_adv_fgm.float()).mean().item())
    df_accs.append((1 - is_adv_df.float()).mean().item())

    i += 1
    if i == 20:
        break

fgm_acc = np.mean(fgm_accs)
df_acc = np.mean(df_accs)
clean_acc = np.mean(clean_accs)

print (f'FastGRadientMethod: {fgm_acc:.2f}, Deep Fool: {df_acc:.2f}, Clean Acc: {clean_acc:.2f}')

fout = open(args.weight_path.replace('.pt', '_attack.txt'), 'w')
fout.write(f'Clean Accuracy: {clean_acc:.2f}\n')
fout.write(f'FGM Accuracy: {fgm_acc:.2f}\n')
fout.write(f'DeepFool Accuracy: {df_acc:.2f}\n')
fout.close()
