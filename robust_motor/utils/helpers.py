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
    inp_channels = 3
    out_channels = 3
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

    print ('Parameters :', sum(p.numel() for p in model.parameters()))

    return model.cuda(args.gpu)

def get_loss_function(args):
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    if args.loss == 'sc_mse':
        criterion = sc_mse

    return criterion

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