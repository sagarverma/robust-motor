import torch
import torch.nn as nn


class ACNN(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, out_channels=256, att_channels=256, n_len_seg=1000, n_classes=5):
        super(ACNN, self).__init__()
        
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_channels = att_channels

        self.cnn1 = nn.Conv1d(in_channels=self.in_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=16, 
                            stride=4)
        self.cnn2 = nn.Conv1d(in_channels=self.out_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=8, 
                            stride=2)
        self.cnn3 = nn.Conv1d(in_channels=self.out_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=4, 
                            stride=1)

        self.W_att_channel = nn.Parameter(torch.randn(self.out_channels, self.att_channels))
        self.v_att_channel = nn.Parameter(torch.randn(self.att_channels, 1))

        self.dense1 = nn.Linear(out_channels, self.out_channels)
        self.dense2 = nn.Linear(out_channels, self.n_classes)

        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):

        self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
        assert (self.n_length % self.n_len_seg == 0), "Input n_length should divided by n_len_seg"
        self.n_seg = self.n_length // self.n_len_seg

        out = x
        out = out.permute(0,2,1)
        out = out.view(-1, self.n_len_seg, self.n_channel)
        out = out.permute(0,2,1)
        out = self.dropout(self.cnn1(out))
        out = self.dropout(self.cnn2(out))
        out = self.cnn3(out)
        out = out.mean(-1)
        out = out.view(-1, self.n_seg, self.out_channels)
        e = torch.matmul(out, self.W_att_channel)
        e = torch.matmul(torch.tanh(e), self.v_att_channel)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        gama = torch.div(n1, n2)
        out = torch.sum(torch.mul(gama, out), 1)
        out = self.dropout(self.dense1(out))
        out = self.dense2(out)
        return out
