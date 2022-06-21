import tqdm
import argparse as ag
import tqdm

import foolbox as fb
import torch 
import torch.nn as nn 

from robust_motor.utils.helpers import get_model
from robust_motor.datasets.temperature import get_loaders


parser = ag.ArgumentParser(description='Attack params')

parser.add_argument('--weight_path', type=str, required=True)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--attack', action='store_true')

args = parser.parse_args()
    
dataset_name = args.weight_path.split('/')[-3]
model_name = args.weight_path.split('/')[-2]

args.model = model_name
args.dataset = dataset_name 

_, val_loader = get_loaders(args)

model = get_model(args)
weight = torch.load(args.weight_path, map_location=args.gpu)
model.load_state_dict(weight)
model.eval()

fmodel = fb.PyTorchModel(model, bounds=(0, 1))

attack = fb.attacks.LinfDeepFoolAttack()

for X, y in tqdm.tqdm(val_loader):
    X = X.to(args.gpu)
    y = y.to(args.gpu)
    raw, clipped, is_adv = attack(fmodel, X, y, epsilons=args.eps)
    print (is_adv.shape)
    break