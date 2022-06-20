import tqdm
import glob
import random 

import numpy as np 
import scipy.io as sio

import torch.utils.data as data 

quantities_min_max = {'current_a': [-18.15417, 18.25],
                    'current_b': [-18.55, 18.43333],
                    'current_c': [-18.24167, 18.34167],
                    'voltage_a': [-250.7333, 250.1917],
                    'voltage_b': [-250.525, 250.0],
                    'voltage_c': [-250.6917, 250.0667],
                    'trigger': [-0.22306311591250358, 6.286101529740748],
                    'vib_acpe': [-58.42140061846215, 49.12634656322328],
                    'vib_acpi': [-82.19557364067303, 68.43362694025562],
                    'vib_axial': [-4.012557587132455, 3.857262295784517],
                    'vib_base': [-100.0, 94.1490775375713],
                    'vib_carc': [-76.13836480814669, 71.55926727574153],
                    'torque': [0.5, 4.0]
                    }

def normalize(data, quantity):
    """Normalize a quantity using global minima and maxima.
    Args:
        data (np.array): Electrical motor quantity as np.array.
        quantity (str): Name of the quantity
    Returns:
        np.array: Normalized electrical motor quantity.
    Raises:        ExceptionName: Why the exception is raised.
    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>
    """
    if data.max() > quantities_min_max[quantity][1] or \
        data.min() < quantities_min_max[quantity][0]:
        print (quantity, data.max(), data.min())
    a = 0
    b = 1
    minn, maxx = quantities_min_max[quantity]
    if minn > data.min() or maxx < data.max():
        print (quantity, data.min(), data.max())
    t = a + (data - minn) * ((b - a) / (maxx - minn))
    return t.astype(np.float32)


def denormalize(data, quantity):
    """Denormalize a quantity using global minima and maxima.
    Args:
        data (np.array): Normalized electrical motor quantity as np.array.
        quantity (str): Name of the quantity
    Returns:
        np.array: Denormalized electrical motor quantity.
    Raises:        ExceptionName: Why the exception is raised.
    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>
    """
    a, b = quantities_min_max[quantity]
    t = a + (data - (0)) * ((b-a) / (1-(0)))
    return t.astype(np.float32)

def load_data(root):
    exps = glob.glob(f"{root}/*.mat")
    cls_map = {'r1b': 1, 'r2b': 2, 'r3b': 3, 'r4b': 4, 'rs': 0}
    torque_map = {"torque05": 0.5, "torque10": 1.0, "torque15": 1.5, 
                  "torque20": 2.0, "torque25": 2.5, "torque30": 3.0,
                  "torque35": 3.5, "torque40": 4.0}
    dataset = {}
    train_samples = []
    val_samples = []

    print ('Loading Dataset')

    for exp in tqdm.tqdm(exps[:2]):
        data = sio.loadmat(exp)

        current_a = normalize(data['current_a'], 'current_a')
        current_b = normalize(data['current_b'], 'current_b')
        current_c = normalize(data['current_c'], 'current_c')
        voltage_a = normalize(data['voltage_a'], 'voltage_a')
        voltage_b = normalize(data['voltage_b'], 'voltage_b')
        voltage_c = normalize(data['voltage_c'], 'voltage_c')
        trigger = normalize(data['trigger'], 'trigger')
        vib_acpe = normalize(data['vib_acpe'], 'vib_acpe')
        vib_acpi = normalize(data['vib_acpi'], 'vib_acpi')
        vib_axial = normalize(data['vib_axial'], 'vib_axial')
        vib_base = normalize(data['vib_base'], 'vib_base')
        vib_carc = normalize(data['vib_carc'], 'vib_carc')

        name = exp.split('/')[-1].split('.')[0]
        cls, torque, exp_no = name.split('_')
        
        torque = np.ones(vib_carc.shape) * torque_map[torque]
        torque = normalize(torque, 'torque')

        lbl = cls_map[cls]

        dataset[name] =  np.stack([current_a[0], current_b[0], current_c[0],
                                    voltage_a[0], voltage_b[0], voltage_c[0],
                                    trigger[0], vib_acpe[0], vib_acpi[0],
                                    vib_axial[0], vib_base[0], vib_carc[0],
                                    torque[0]], axis=0)

        samples = []
        for i in range(0, current_a.shape[1], 1000):
            if (i + 1000) < current_a.shape[1]:
                samples.append([name, i, i+1000, lbl])

        if int(exp_no) < 7:
            train_samples += samples
        else:
            val_samples += samples

    return dataset, train_samples, val_samples


class BrokernBarsLoader(data.Dataset):
    def __init__(self, full_load, samples):
        random.shuffle(samples)
        self.samples = samples 
        self.full_load = full_load 
    
    def __getitem__(self, index):
        name, start, end, lbl = self.samples[index]

        inp_seq = self.full_load[name][:, start:end]

        return inp_seq, lbl 

    def __len__(self):
        return len(self.samples)


def get_loaders(args):
    dataset, train_samples, val_samples = load_data('data/BrokenBars/')

    print('train samples : ', len(train_samples))
    print('val samples : ', len(val_samples))

    train_preloader = BrokernBarsLoader(dataset, train_samples)
    val_preloader = BrokernBarsLoader(dataset, val_samples)

    train_loader = data.DataLoader(train_preloader, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_preloader, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader