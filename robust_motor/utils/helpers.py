import os

import torch
import torch.nn as nn

from robust_motor.utils.metrics import sc_mse

from robust_motor.models.cnn import ShallowCNN, DeepCNN
from robust_motor.models.ffnn import ShallowFNN, DeepFNN
from robust_motor.models.rnn import ShallowRNN, DeepRNN
from robust_motor.models.lstm import ShallowLSTM, DeepLSTM
from robust_motor.models.encdec import (ShallowEncDec, DeepEncDec, EncDecSkip,
                          EncDecRNNSkip, EncDecBiRNNSkip,
                          EncDecDiagBiRNNSkip)
from robust_motor.models.resnet1d import ResNet1D
from robust_motor.models.crnn1d import CRNN
from robust_motor.models.acnn1d import ACNN
from robust_motor.models.regnet1d import RegNet1D
from robust_motor.models.transformer1d import Transformer1D
from robust_motor.models.fedformer import FedFormer


def get_paths(args):
    """Get file fully qualified names to write weights and logs.
    Args:
        args (argparse.ArgumentParser): Parsed arguments.
    Returns:
        tuple: weight path and log path.
    Raises:        ExceptionName: Why the exception is raised.
    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>
    """

    weight_dir = f"weights/{args.dataset}/{args.model}"
    log_dir = f"logs/{args.dataset}/{args.model}"

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return weight_dir, log_dir


def get_model(args):
    """Get model.
    Args:
        args (argparse.ArgumentParser): Parsed arguments.
    Returns:
        torch.nn.module: Model definition.
    Raises:        ExceptionName: Why the exception is raised.
    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>
    """
    if args.dataset == 'MotorDynamics':
        inp_channels = 3
        out_channels = 3
    if args.dataset == 'MotorDenoise':
        inp_channels = 4
        out_channels = 4
    if args.dataset == 'SpeedTorque':
        inp_channels = 4
        out_channels = 2
    if args.dataset == 'BrokenBars':
        inp_channels = 13
        num_classes = 5
    if args.dataset == 'Temperature':
        inp_channels = 11
        out_channels = 1

    act = 'relu'

    if args.model == 'shallow_fnn':
        inp_len = inp_channels * 100
        model = ShallowFNN(inp_len, out_channels, act)
    if args.model == 'deep_fnn':
        inp_len = inp_channels * 100
        model = DeepFNN(inp_len, out_channels, act)
    if args.model == 'shallow_cnn':
        model = ShallowCNN(inp_channels, out_channels, act)
    if args.model == 'deep_cnn':
        model = DeepCNN(inp_channels, out_channels, act)
    if args.model == 'shallow_rnn':
        model = ShallowRNN(inp_channels, out_channels, 32, act)
    if args.model == 'deep_rnn':
        model = DeepRNN(inp_channels, out_channels, 32, act)
    if args.model == 'shallow_lstm':
        model = ShallowLSTM(inp_channels, out_channels, 32, act)
    if args.model == 'deep_lstm':
        model = DeepLSTM(inp_channels, out_channels, 32, act)
    if args.model == 'shallow_encdec':
        model = ShallowEncDec(inp_channels, out_channels, act)
    if args.model == 'deep_encdec':
        model = DeepEncDec(inp_channels, out_channels, act)
    if args.model == 'encdec_skip':
        model = EncDecSkip(inp_channels, out_channels, act)
    if args.model == 'encdec_rnn_skip':
        model = EncDecRNNSkip(inp_channels, out_channels, act)
    if args.model == 'encdec_birnn_skip':
        model = EncDecBiRNNSkip(inp_channels, out_channels, act)
    if args.model == 'encdec_diag_birnn_skip':
        model = EncDecDiagBiRNNSkip(inp_channels, out_channels, act)
    if args.model == 'resnet1d':
        model = ResNet1D(in_channels=inp_channels, n_classes=num_classes,
                        n_block=16)
    if args.model == 'crnn1d':
        model = CRNN(in_channels=inp_channels, n_classes=num_classes)
    if args.model == 'acnn1d':
        model = ACNN(in_channels=inp_channels, n_classes=num_classes)
    if args.model == 'regnet1d':
        model = RegNet1D(in_channels=inp_channels, n_classes=num_classes)
    if args.model == 'transformer1d':
        model = Transformer1D(inp_channels=inp_channels, n_classes=num_classes)
    if args.model == 'fedformer':
        model = FedFormer(enc_in=inp_channels, dec_in=inp_channels, 
                            c_out=out_channels, seq_len=1000, label_len=1000, pred_len=1000)
                            
    print ('Parameters :', sum(p.numel() for p in model.parameters()))

    return model.cuda(args.gpu)

def get_model_from_weight(args):
    model = torch.load(args.weight_file)
    return model


def log(
        writer, epoch, split, loss, metrics
    ):
    print(f'Epoch {(epoch+1):02d} {split.capitalize()} Loss {loss:.4f}')
    writer.add_scalar(f'{split}/loss', loss, epoch)

    for k in metrics.keys():
        print(f'{k} {metrics[k]:.4f}')
        writer.add_scalar(f'{split}/{k}', metrics[k], epoch)