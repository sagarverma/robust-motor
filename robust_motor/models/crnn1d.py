import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, in_channels, out_channels=256, n_len_seg=1000, n_classes=5):
        super(CRNN, self).__init__()
        
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # (batch, channels, length)
        self.cnn = nn.Conv1d(in_channels=self.in_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=16, 
                            stride=2)
        # (batch, seq, feature)
        self.rnn = nn.LSTM(input_size=(self.out_channels), 
                            hidden_size=self.out_channels, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=False)
        self.dense = nn.Linear(out_channels, n_classes)
        
    def forward(self, x):

        self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
        assert (self.n_length % self.n_len_seg == 0), "Input n_length should divided by n_len_seg"
        self.n_seg = self.n_length // self.n_len_seg

        out = x
        out = out.permute(0,2,1)
        out = out.view(-1, self.n_len_seg, self.n_channel)
        out = out.permute(0,2,1)
        out = self.cnn(out)
        out = out.mean(-1)
        out = out.view(-1, self.n_seg, self.out_channels)
        _, (out, _) = self.rnn(out)
        out = torch.squeeze(out, dim=0)
        out = self.dense(out)        
        return out