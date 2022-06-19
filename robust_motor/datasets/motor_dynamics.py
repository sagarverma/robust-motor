import os
import random

import numpy as np
import scipy.io as sio

import torch.utils.data as data



quantities_min_max = {'voltage_d': (-500, 500),
                      'voltage_q': (-500, 500),
                      'speed': (-700, 700),
                      'current_d': (-30, 30),
                      'current_q': (-30, 30),
                      'torque': (-250, 250)}


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
    """Load synthetic dataset.
    Args:
        root (type): Dataset directory to load the dataset.
    Returns:
        tuple: dataset, index_quant_map.
    Raises:        ExceptionName: Why the exception is raised.
    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>
    """
    exps = os.listdir(root)

    dataset = []
    for exp in exps:
        mat_data, index_quant_map = _load_exp_data(os.path.join(root, exp))
        dataset.append(mat_data)

    return dataset, index_quant_map


def _load_exp_data(root):
    data = sio.loadmat(root)

    noisy_voltage_d = normalize(data['noisy_voltage_d'], 'voltage_d')
    voltage_d = normalize(data['voltage_d'][0, :], 'voltage_d')
    voltage_q = normalize(data['voltage_q'][0, :], 'voltage_q')
    noisy_voltage_q = normalize(data['noisy_voltage_q'][0, :], 'voltage_q')
    current_d = normalize(data['current_d'][0, :], 'current_d')
    noisy_current_d = normalize(data['noisy_current_d'][0, :], 'current_d')
    current_q = normalize(data['current_q'][0, :], 'current_q')
    noisy_current_q = normalize(data['noisy_current_q'][0, :], 'current_q')
    speed = normalize(data['speed'][0, :], 'speed')
    torque = normalize(data['torque'][0, :], 'torque')
    time = data['time'][0, :]


    dataset = (noisy_voltage_d, voltage_d, noisy_voltage_q, voltage_q, speed,
               noisy_current_d, current_d, noisy_current_q, current_q, torque, time)
    dataset = np.vstack(dataset)

    index_quant_map = {'noisy_voltage_d': 0, 
                       'voltage_d': 1,
                       'noisy_voltage_q': 2,
                       'voltage_q': 3,
                       'speed': 4,
                       'noisy_current_d': 5,
                       'current_d': 6,
                       'noisy_current_q': 7,
                       'current_q': 8,
                       'torque': 9,
                       'time': 10}

    return dataset.astype(np.float32), index_quant_map


def get_sample_metadata(dataset, stride=1, window=100):
    """Get sample metadata from dataset based on sampling stride and window.
    Args:
        dataset (list): List of np.array extracted from different mat files.
        stride (int): Sampling stride.
        window (int): Sampling window length.
    Returns:
        list: List of samples where each item in list is a tuple with
              mat no, index in mat data, index + window and index + window//2.
    Raises:        ExceptionName: Why the exception is raised.
    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>
    """
    samples = []

    for sample_no in range(len(dataset)):
        for i in range(0, dataset[sample_no].shape[1], stride):
            if i + window < dataset[sample_no].shape[1]:
                samples.append([sample_no, i, i+window, i+window//2])

    return samples


class FlatInFlatOut(data.Dataset):
    def __init__(self, full_load, index_quant_map, samples,
                 inp_quants, out_quants):
        """Dataloader class to load samples from signals loaded.
        Args:
            full_load (list): List of numpy array of loaded mat files.
            index_quant_map (dict): Dictionary which maps signal quantity to
                                    to index in full_load arrays.
            samples (list): Metadata used to sample subsequences from
                            full_load.
            inp_quants (list): Input quantities to the model.
            out_quants (list): Output quantities to the model.
        Returns:
            type: Description of returned object.
        Raises:            ExceptionName: Why the exception is raised.
        Examples
            Examples should be written in doctest format, and
            should illustrate how to use the function/class.
            >>>
        """
        random.shuffle(samples)
        self.samples = samples
        self.full_load = full_load
        self.inp_quant_ids = [index_quant_map[x] for x in inp_quants]
        self.out_quant_ids = [index_quant_map[x] for x in out_quants]

    def __getitem__(self, index):
        mat_no, start, end, infer_index = self.samples[index]

        inp_seq = self.full_load[mat_no][self.inp_quant_ids, start: end]
        out_seq = self.full_load[mat_no][self.out_quant_ids, infer_index]
        inp_seq = inp_seq.flatten()
        out_seq = out_seq.flatten()

        return inp_seq, out_seq

    def __len__(self):
        return len(self.samples)


class SeqInFlatOut(data.Dataset):
    def __init__(self, full_load, index_quant_map, samples,
                 inp_quants, out_quants):
        """Dataloader class to load samples from signals loaded.
        Args:
            full_load (list): List of numpy array of loaded mat files.
            index_quant_map (dict): Dictionary which maps signal quantity to
                                    to index in full_load arrays.
            samples (list): Metadata used to sample subsequences from
                            full_load.
            inp_quants (list): Input quantities to the model.
            out_quants (list): Output quantities to the model.
        Returns:
            type: Description of returned object.
        Raises:            ExceptionName: Why the exception is raised.
        Examples
            Examples should be written in doctest format, and
            should illustrate how to use the function/class.
            >>>
        """
        random.shuffle(samples)
        self.samples = samples
        self.full_load = full_load
        self.inp_quant_ids = [index_quant_map[x] for x in inp_quants]
        self.out_quant_ids = [index_quant_map[x] for x in out_quants]

    def __getitem__(self, index):
        mat_no, start, end, infer_index = self.samples[index]

        inp_seq = self.full_load[mat_no][self.inp_quant_ids, start: end]
        out_seq = self.full_load[mat_no][self.out_quant_ids, infer_index]
        out_seq = out_seq.flatten()

        return inp_seq, out_seq

    def __len__(self):
        return len(self.samples)


class SeqInSeqOut(data.Dataset):
    def __init__(self, full_load, index_quant_map, samples,
                 inp_quants, out_quants):
        """Dataloader class to load samples from signals loaded.
        Args:
            full_load (list): List of numpy array of loaded mat files.
            index_quant_map (dict): Dictionary which maps signal quantity to
                                    to index in full_load arrays.
            samples (list): Metadata used to sample subsequences from
                            full_load.
            inp_quants (list): Input quantities to the model.
            out_quants (list): Output quantities to the model.
        Returns:
            type: Description of returned object.
        Raises:            ExceptionName: Why the exception is raised.
        Examples
            Examples should be written in doctest format, and
            should illustrate how to use the function/class.
            >>>
        """
        random.shuffle(samples)
        self.samples = samples
        self.full_load = full_load
        self.inp_quant_ids = [index_quant_map[x] for x in inp_quants]
        self.out_quant_ids = [index_quant_map[x] for x in out_quants]

    def __getitem__(self, index):
        mat_no, start, end, _ = self.samples[index]

        inp_seq = self.full_load[mat_no][self.inp_quant_ids, start: end]
        out_seq = self.full_load[mat_no][self.out_quant_ids, start: end]

        return inp_seq, out_seq

    def __len__(self):
        return len(self.samples)


def _get_prelaoder_class(args):
    if 'fnn' in args.model:
        return FlatInFlatOut
    if 'cnn' in args.model:
        return SeqInFlatOut
    if 'rnn' in args.model or 'lstm' in args.model or 'encdec' in args.model:
        return SeqInSeqOut


def _get_loader(dir, args, shuffle):
    dataset, index_quant_map = load_data(dir)
    samples = get_sample_metadata(dataset, 1, 100)
    preloader_class = _get_prelaoder_class(args)
    if args.dataset == 'MotorDynamics':
        inp_quants = ['voltage_d', 'voltage_q', 'speed']
        out_quants = ['current_d', 'current_q', 'torque']
    if args.dataset == 'MotorDenoise':
        inp_quants = ['noisy_voltage_d', 'noisy_voltage_q', 'noisy_current_d', 'noisy_current_q']
        out_quants = ['voltage_d', 'voltage_q', 'current_d', 'current_q']
    if args.dataset == 'SpeedTorque':
        inp_quants = ['voltage_d', 'voltage_q', 'current_d', 'current_q']
        out_quants = ['speed', 'torque']

    preloader = preloader_class(dataset, index_quant_map, samples,
                                inp_quants, out_quants)
    dataloader = data.DataLoader(preloader, batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=args.num_workers)
    return dataloader, len(samples)


def get_loaders(args):
    """Get dataloaders for training, and validation.
    Args:
        args (argparse.ArgumentParser): Parsed arguments.
    Returns:
        tuple: train sim dataloader and val sim dataloader
    Raises:        ExceptionName: Why the exception is raised.
    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>
    """

    train_loader, train_samples = _get_loader('data/Data_27012021_noisy/train', args, True)
    val_loader, val_samples = _get_loader('data/Data_27012021_noisy/val', args, False)

    print('train samples : ', train_samples)
    print('val samples : ', val_samples)

    return train_loader, val_loader