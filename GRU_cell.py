import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class GRUCell(nn.Module):
    """
    An implementation of GRUCell.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Sequential(
            nn.Linear(input_size, 3*hidden_size, bias=True),
            #nn.ReLU(),nn.Linear(hidden_size, 3* hidden_size)

        )
        #self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        #self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Sequential(
            nn.Linear(hidden_size, 3*hidden_size, bias=True),
            #nn.ReLU(),nn.Linear(hidden_size, 3* hidden_size)
        )
        self.reset_parameters()

    def reset_parameters(self):
        '''
        std = 0.1 #1.0 / math.sqrt(self.hidden_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Change nonlinearity to 'leaky_relu' if you switch
                nn.init.normal(m.weight, std=0.1)
                nn.init.constant_(m.bias, 0.1)
        '''
        #std = 1.0 / math.sqrt(self.hidden_size)
        std = 0.1
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, x, hidden, w=None, w1=None):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        #gate_x = gate_x.squeeze()
        #gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        if w is None:
            inputgate = F.sigmoid(i_i + h_i)
        else:
            #inputgate = 1 - (1-F.sigmoid(i_i + h_i)) * w
            inputgate = F.sigmoid(i_i + h_i) *w
        if w1 is None:
            newgate = F.tanh(i_n + (resetgate * h_n))
        else:
            newgate = F.tanh(i_n + (resetgate * h_n)) * w1

        hy = newgate + inputgate * (hidden - newgate)

        return hy

def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True, dropconnect=0.):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        #gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.hidden_size).zero_())


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size, temperature=1.0, dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)
        self.temperature = temperature

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True)
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size*4+self.n_chunk*2, bias=True, dropout=dropconnect)

        self.drop_weight_modules = [self.hh]
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hidden,
                transformed_input=None):
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input)

        gates = transformed_input + self.hh(hx)
        cingate, cforgetgate = gates[:, :self.n_chunk*2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:,self.n_chunk*2:].view(-1, self.n_chunk*4, self.chunk_size).chunk(4,1)

        cingate = cumsoftmax(cingate)
        cingate = 1. - cingate
        cforgetgate = cumsoftmax(cforgetgate)

        distance_cforget = 1. - cforgetgate.sum(dim=-1) / self.n_chunk
        distance_cin = cingate.sum(dim=-1) / self.n_chunk

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cell = F.tanh(cell)
        outgate = F.sigmoid(outgate)

        #cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        #hy = outgate * F.tanh(self.c_norm(cy))
        hy = outgate * F.tanh(cy)
        return hy.view(-1, self.hidden_size), cy, (distance_cforget, distance_cin)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_())

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()
