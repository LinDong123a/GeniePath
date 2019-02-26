import torch

class AdaptiveBeadthFunc(torch.nn.Module):
    """Using Attention Mechanism for gathering infomation of a node from its neigbhours
    
    Attributes:
    iputs:      n_node*feature_dim tensor, requiring for getting the whole graph node
                infomation for getting information from neighbours
    out_dim:    the output dims
    bias_mtx:   Mask for attention mechanism
    """
    def __init__(self, in_dim, out_dim, attn_dropout=0.0, ff_dropout=0.0,
                 activate=torch.tanh, normalize=torch.softmax):
        super(AdaptiveBeadthFunc, self).__init__()
        
        self.in_dim = in_dim
        self.output_dim = out_dim
        self.out_dim = out_dim

        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        
        self.Ws = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.Wd = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.W = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.v = torch.nn.Linear(in_dim, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.Ws.weight)
        torch.nn.init.xavier_uniform_(self.Wd.weight)
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.v.weight)

        self.activate = activate
        self.normalize = normalize

    def forward(self, inputs, bias_mtx=None, training=True):
        n_node, dim = inputs.shape

        inputs = torch.nn.functional.dropout(inputs, p=self.ff_dropout ,training=training)

        """
        f_1 = self.v(torch.tanh(self.Ws(inputs)))
        f_2 = self.v(torch.tanh(self.Wd(inputs)))
        """
        f_1 = self.v(self.Ws(inputs))
        f_2 = self.v(self.Wd(inputs))

        f = f_1 + torch.transpose(f_2, 0, 1)
        coef = torch.tanh(f)

        if bias_mtx is None:
            coef = self.normalize(coef, dim=-1)
        else:
            coef = self.normalize(coef + bias_mtx, dim=-1)

        coef = torch.nn.functional.dropout(coef, p=self.attn_dropout, training=training)

        ret = self.W(torch.mm(coef, inputs))

        return self.activate(ret)
        
class AdaptiveDepthFunc(torch.nn.Module):
    """Using LSTM for training
    Attributes:
        iputs:      n_node*feature_dim tensor, requiring for getting the whole graph node
                    infomation for getting information from neighbours
        states:     should be in the same shape with inputs

    Returns:
        next_states: same shape with states
        ret:
    """
    def __init__(self, dim, ff_dropout=0.0):
        super(AdaptiveDepthFunc, self).__init__()
        self.ff_dropout = ff_dropout

        self.Wf = torch.nn.Linear(dim, dim, bias=False)
        self.Wi = torch.nn.Linear(dim, dim, bias=False)
        self.Wo = torch.nn.Linear(dim, dim, bias=False)
        self.Wc = torch.nn.Linear(dim, dim, bias=False)

        torch.nn.init.xavier_uniform_(self.Wf.weight)
        torch.nn.init.xavier_uniform_(self.Wi.weight)
        torch.nn.init.xavier_uniform_(self.Wo.weight)
        torch.nn.init.xavier_uniform_(self.Wc.weight)

    def forward(self, inputs, states, training=True):
        inputs = torch.nn.functional.dropout(inputs, p=self.ff_dropout, training=training)

        ii = torch.sigmoid(self.Wf(inputs))
        fi = torch.sigmoid(self.Wi(inputs))
        oi = torch.sigmoid(self.Wo(inputs))
        C_head = torch.tanh(self.Wc(inputs))

        next_states = fi*states + ii*C_head
        ret = torch.tanh(next_states) * oi

        return next_states, ret
