import tqdm
import glob 
import random 

import numpy as np 
import pandas as pd

import torch.utils.data as data


quantities_min_max = {'i_d': [-278.00, 0.05],
                    'i_q': [-293.43, 301.71],
                    'u_d': [-131.53, 131.47], 
                    'u_q': [-25.29, 133.04], 
                    'motor_speed': [-275.55, 6000.02],
                    'torque': [-246.47, 261.01],
                    'coolant': [10.62, 101.60], 
                    'ambient': [8.78, 30.71],
                    'stator_winding': [18.59, 141.36],
                    'stator_tooth': [18.13, 111.95],  
                    'stator_yoke': [18.08, 100.52],
                    'pm': [20.86, 113.61]
                    }


def normalize(data, quantity):
    # if data.max() > quantities_min_max[quantity][1] or \
    #     data.min() < quantities_min_max[quantity][0]:
        # print (quantity, data.max(), data.min())
    a = 0
    b = 1
    minn, maxx = quantities_min_max[quantity]
    # if minn > data.min() or maxx < data.max():
    #     print (quantity, data.min(), data.max())
    t = a + (data - minn) * ((b - a) / (maxx - minn))
    return t.astype(np.float32)

def normalize_quants(profile_df):
    i_d = normalize(profile_df['i_d'], 'i_d')
    i_q = normalize(profile_df['i_q'], 'i_q')
    u_d = normalize(profile_df['u_d'], 'u_d')
    u_q = normalize(profile_df['u_q'], 'u_q')
    motor_speed = normalize(profile_df['motor_speed'], 'motor_speed')
    torque = normalize(profile_df['torque'], 'torque')
    coolant = normalize(profile_df['coolant'], 'coolant')
    ambient = normalize(profile_df['ambient'], 'ambient')
    stator_winding = normalize(profile_df['stator_winding'], 'stator_winding')
    stator_tooth = normalize(profile_df['stator_tooth'], 'stator_tooth')
    stator_yoke = normalize(profile_df['stator_yoke'], 'stator_yoke')
    pm = normalize(profile_df['pm'], 'pm')

    out = np.stack([i_d, i_q, u_d,
                    u_q, motor_speed, torque,
                    coolant, ambient, stator_winding,
                    stator_tooth, stator_yoke, pm])
    # print (out.shape)
    return out

def load_data(root):
    train_df = pd.read_csv(f'{root}/train.csv')
    val_df = pd.read_csv(f'{root}/val.csv')
    train_dataset = {}
    val_dataset = {}
    train_samples = []
    val_samples = []

    print ('Loading Dataset')

    for profile_id in tqdm.tqdm(train_df.profile_id.unique()):
        profile_df = train_df[train_df['profile_id'] == profile_id]

        train_dataset[profile_id] =  normalize_quants(profile_df)

        for i in range(0, profile_df.shape[0], 10):
            if (i + 1000) < profile_df.shape[0]:
                train_samples.append([profile_id, i, i+1000])

    for profile_id in tqdm.tqdm(val_df.profile_id.unique()):
        profile_df = val_df[val_df['profile_id'] == profile_id]

        val_dataset[profile_id] =  normalize_quants(profile_df)

        for i in range(0, profile_df.shape[0], 1):
            if (i + 1000) < profile_df.shape[0]:
                val_samples.append([profile_id, i, i+1000])
        
    return train_dataset, val_dataset, train_samples, val_samples


class TemperatureLoader(data.Dataset):
    def __init__(self, full_load, samples):
        random.shuffle(samples)
        self.samples = samples 
        self.full_load = full_load 
    
    def __getitem__(self, index):
        name, start, end = self.samples[index]

        inp_seq = self.full_load[name][:-1, start:end]
        out_seq = self.full_load[name][-2:-1, start:end]
        return inp_seq, out_seq

    def __len__(self):
        return len(self.samples)

def get_loaders(args):
    train_dataset, val_dataset, train_samples, val_samples = load_data('data/Temperature/')

    print('train samples : ', len(train_samples))
    print('val samples : ', len(val_samples))

    train_preloader = TemperatureLoader(train_dataset, train_samples)
    val_preloader = TemperatureLoader(val_dataset, val_samples)

    train_loader = data.DataLoader(train_preloader, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_preloader, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader