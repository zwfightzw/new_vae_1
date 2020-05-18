import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, kl_divergence


class Flatten(nn.Module):
    def forward(self, input_data):
        if len(input_data.size()) == 4:
            return input_data.view(input_data.size(0), -1)
        else:
            return input_data.view(input_data.size(0), input_data.size(1), -1)


class LinearLayer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 nonlinear=nn.ELU(inplace=True)):
        super(LinearLayer, self).__init__()
        # linear
        self.linear = nn.Linear(in_features=input_size,
                                out_features=output_size)

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.linear(input_data))


class ConvLayer1D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvLayer1D, self).__init__()
        # linear
        self.linear = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm1d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class ConvLayer2D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvLayer2D, self).__init__()
        # linear
        self.linear = nn.Conv2d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm2d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class ConvTransLayer2D(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 normalize=True,
                 nonlinear=nn.ELU(inplace=True)):
        super(ConvTransLayer2D, self).__init__()
        # linear
        self.linear = nn.ConvTranspose2d(in_channels=input_size,
                                         out_channels=output_size,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False if normalize else True)
        if normalize:
            self.normalize = nn.BatchNorm2d(num_features=output_size)
        else:
            self.normalize = nn.Identity()

        # nonlinear
        self.nonlinear = nonlinear

    def forward(self, input_data):
        return self.nonlinear(self.normalize(self.linear(input_data)))


class Encoder(nn.Module):
    def __init__(self,
                 output_size=None,
                 feat_size=64,
                 channel=1, shape=64):
        super(Encoder, self).__init__()
        if shape==64:
            network_list = [ConvLayer2D(input_size=channel,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 32 x 32
                        ConvLayer2D(input_size=feat_size,output_size=feat_size,kernel_size=4,stride=2,padding=1),  # 16 x 16
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 8 x 8
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 4 x 4
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0),  # 1 x 1
                        Flatten()]
        else:
            network_list = [ConvLayer2D(input_size=channel,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 32 x 32
                        #ConvLayer2D(input_size=feat_size,output_size=feat_size,kernel_size=4,stride=2,padding=1),  # 16 x 16
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 8 x 8
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1),  # 4 x 4
                        ConvLayer2D(input_size=feat_size,
                                    output_size=feat_size,
                                    kernel_size=4,
                                    stride=1,
                                    padding=0),  # 1 x 1
                        Flatten()]
        if output_size is not None:
            network_list.append(LinearLayer(input_size=feat_size,
                                            output_size=output_size))
            self.output_size = output_size
        else:
            self.output_size = feat_size

        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        return self.network(input_data)


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 feat_size=64,
                 channel=1, dataset='moving_mnist', shape=64):
        super(Decoder, self).__init__()
        if input_size == feat_size:
            self.linear = nn.Identity()
        else:
            self.linear = LinearLayer(input_size=input_size,
                                      output_size=feat_size,
                                      nonlinear=nn.Identity())
        if dataset == 'moving_mnist' or dataset == 'bouncing_balls':
            nolinear_dataset = nn.Sigmoid()
        elif dataset == 'lpc':
            nolinear_dataset = nn.Tanh()
        if shape==64:
            self.network = nn.Sequential(ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size,
                                                      kernel_size=4,
                                                      stride=1,
                                                      padding=0),
                                     ConvTransLayer2D(input_size=feat_size,output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=feat_size),
                                     ConvTransLayer2D(input_size=feat_size,
                                                      output_size=channel,
                                                      normalize=False,
                                                      nonlinear=nolinear_dataset))
        else:
            self.network = nn.Sequential(ConvTransLayer2D(input_size=feat_size,
                                                          output_size=feat_size,
                                                          kernel_size=4,
                                                          stride=1,
                                                          padding=0),
                                         # ConvTransLayer2D(input_size=feat_size,output_size=feat_size),
                                         ConvTransLayer2D(input_size=feat_size,
                                                          output_size=feat_size),
                                         ConvTransLayer2D(input_size=feat_size,
                                                          output_size=feat_size),
                                         ConvTransLayer2D(input_size=feat_size,
                                                          output_size=channel,
                                                          normalize=False,
                                                          nonlinear=nolinear_dataset))
    def forward(self, input_data):
        return self.network(self.linear(input_data).unsqueeze(-1).unsqueeze(-1))

