import torch

from layers import (
    AdaptiveBeadthFunc, AdaptiveDepthFunc)

class GeniePath(torch.nn.Module):
    """Implement for pytorch geniepath model
    Attributes:
        hidden_units: list, specify the output dim of each hidden layer
        in_dim: number of features as input
        hidden_units: considering the structure of the network, the hidden units is shared
                    across layer, so it should only an int
    """
    def __init__(self, in_dim, n_class, hidden_units, n_layer, 
                 attn_dropout=0.0, ff_dropout=0.0):
        super(GeniePath, self).__init__()
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.in_dim = in_dim
        self.n_class = n_class
        self.hidden_units = hidden_units
        self.n_layer = n_layer

        self.Wx = torch.nn.Linear(in_dim, hidden_units)
        self.hidden2class = torch.nn.Linear(hidden_units, n_class, bias=False)

        torch.nn.init.xavier_uniform_(self.Wx.weight)
        torch.nn.init.xavier_uniform_(self.hidden2class.weight)
        
        self.breadth_layers = []
        for _ in range(n_layer):
            self.breadth_layers.append(
                AdaptiveBeadthFunc(hidden_units, hidden_units, 
                                   attn_dropout=self.attn_dropout,
                                   ff_dropout=self.ff_dropout))

        self.depth_layers = []
        for _ in range(n_layer):
            self.depth_layers.append(AdaptiveDepthFunc(hidden_units,
                                                       ff_dropout=self.ff_dropout))

    def forward(self, inputs, n_node, mask, training=True, bias_mtx=None):
        states = torch.zeros(n_node, self.hidden_units)
        inputs = torch.nn.functional.dropout(inputs, p=self.ff_dropout, training=training)

        # Use Adpative Layer for exploring
        h = self.Wx(inputs)
        for layer_idx in range(self.n_layer):
            h_tmp = self.breadth_layers[layer_idx](h, bias_mtx, training=training)
            states, h = self.depth_layers[layer_idx](h_tmp, states, training=training)

        self.scores = self.hidden2class(h)

        return self.scores

    def predict(self):
        with torch.no_grad():
            return torch.softmax(self.scores, dim=-1)

    def masked_accu(self, pred, label, mask):
        label = label[mask]
        pred = pred[mask, :]
        
        pred_idx = torch.argmax(pred, dim=-1)
        
        n_correct = (pred_idx == label).sum()
        n_tot = pred.shape[0]

        return n_correct.item() / n_tot

