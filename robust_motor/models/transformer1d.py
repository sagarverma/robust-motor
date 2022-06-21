import torch
import torch.nn as nn

    
class Transformer1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes)
        
    Pararmetes:
        
    """

    def __init__(self, inp_channels, n_classes=5, n_length=1000, d_model=64, nhead=8, dim_feedforward=64, dropout=0.1):
        super(Transformer1D, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_length = n_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.embedding_layer = nn.Linear(inp_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense = nn.Linear(self.d_model, self.n_classes)
        
    def forward(self, x):
        out = x
        out = out.permute(2, 0, 1)
        out = self.embedding_layer(out)
        out = self.transformer_encoder(out)
        out = out.mean(0)
        out = self.dense(out)
        
        return out