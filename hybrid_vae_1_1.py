import os
import torchvision
import torch
import torch.utils.data
import torch.nn.init
import torch.optim as optim
from torch.autograd import Variable
from GRU_cell import GRUCell, ONLSTMCell, LSTMCell
from modules import *
import numpy as np
import datetime
import dateutil.tz
import argparse
import data
import torchvision.transforms as transforms
import data.video_transforms as vtransforms


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def concat(*data_list):
    return torch.cat(data_list, 1)


class Sprites(torch.utils.data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.load(self.path + '/%d.sprite' % (idx + 1))


class bouncing_balls(torch.utils.data.Dataset):
    def __init__(self, path, size):
        self.path = path
        self.length = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = np.load(self.path + '/%d.npy' % (idx))

        return torch.from_numpy(data)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, latent_dim, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim)


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.0):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class FullQDisentangledVAE(nn.Module):
    def __init__(self, temperature, frames, z_dim, conv_dim, hidden_dim, block_size, channel, dataset, device, shape):
        super(FullQDisentangledVAE, self).__init__()
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.device = device
        self.dataset = dataset
        self.temperature = temperature
        self.dropout = 0.35

        # self.z_lstm = GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.z_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim // 2, 1, bidirectional=True, batch_first=True)
        # self.z_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, 1, batch_first=True)
        self.z_rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.z_post_out = nn.Linear(self.hidden_dim, self.z_dim * 2)

        # self.z_prior_out_list = nn.Linear(self.hidden_dim,self.z_dim * 2)
        self.z_prior_out_list = nn.Sequential(nn.Linear(self.hidden_dim, self.z_dim * 2))
        self.lockdrop = LockedDropout()
        self.z_to_c_fwd_list = [GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim).to(self.device)
                                for i in range(self.block_size)]

        self.z_w_function = nn.Linear(self.hidden_dim, self.block_size)  # nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim//2),nn.ReLU(), nn.Linear(self.hidden_dim//2, self.block_size))
        # observation encoder / decoder
        self.enc_obs = Encoder(feat_size=self.hidden_dim, output_size=self.hidden_dim, channel=channel, shape=shape)
        self.dec_obs = Decoder(input_size=self.z_dim, feat_size=self.hidden_dim, channel=channel, dataset=dataset,
                               shape=shape)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,
                                        nonlinearity='relu')  #
                # Change nonlinearity to 'leaky_relu' if you switch

            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, std=0.1)
                nn.init.constant_(m.bias, 0.1)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        #random_sampling = True
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def encode_z(self, x):
        batch_size = x.shape[0]
        seq_size = x.shape[1]

        lstm_out, _ = self.z_lstm(x)
        lstm_out, _ = self.z_rnn(lstm_out)
        '''
        lstm_out = x.new_zeros(batch_size, seq_size, self.hidden_dim)
        z_fwd_post = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        for i in range(seq_size):
            z_fwd_post= self.z_lstm(x[:,i],z_fwd_post)
            lstm_out[:, i] = z_fwd_post
        # 
        '''
        each_block_size = self.hidden_dim // self.block_size

        z_post_norm_list = []

        z_prior_norm_list = []
        zt_obs_list = []

        zt_1_post = self.z_post_out(lstm_out[:, 0])
        zt_1_mean = zt_1_post[:, :self.z_dim]
        zt_1_lar = zt_1_post[:, self.z_dim:]

        z_post_norm_list.append(Normal(loc=zt_1_mean, scale=torch.sigmoid(zt_1_lar)))

        z_prior_fwd = self.z_prior_out_list(lstm_out[:, 0])
        z_fwd_latent_mean = z_prior_fwd[:, :self.z_dim]
        z_fwd_latent_lar = z_prior_fwd[:, self.z_dim:]

        # store the prior of ct_i
        z_prior_norm_list.append(Normal(z_fwd_latent_mean, torch.sigmoid(z_fwd_latent_lar)))

        #z_prior_norm_list.append(Normal(torch.zeros(batch_size, self.z_dim).to(self.device), torch.ones(batch_size, self.z_dim).to(self.device)))
        post_z_1 = Normal(zt_1_mean, torch.sigmoid(zt_1_lar)).rsample()
        # init wt
        wt = torch.ones(batch_size, self.block_size).to(self.device)
        z_fwd_list = [torch.zeros(batch_size, self.hidden_dim).to(self.device) for i in range(self.block_size)]
        z_fwd_all = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        # zt_obs = concat(z_fwd_all, post_z_1)

        store_wt = []
        store_wt.append(wt[0].detach().cpu().numpy())
        zt_obs_list.append(post_z_1)

        curr_layer = [None] * (seq_size-1)
        #curr_layer[0] = torch.zeros(batch_size, self.hidden_dim).to(self.device)

        for t in range(1, seq_size):

            z_post_out = self.z_post_out(lstm_out[:, t])
            zt_post_mean = z_post_out[:, :self.z_dim]
            zt_post_lar = z_post_out[:, self.z_dim:]

            z_post_norm_list.append(Normal(zt_post_mean, torch.sigmoid(zt_post_lar)))
            z_post_sample = Normal(zt_post_mean, torch.sigmoid(zt_post_lar)).rsample()


            for fwd_t in range(self.block_size):
                # prior over ct of each block, ct_i~p(ct_i|zt-1_i)

                if fwd_t == 0:
                    zt_1_tmp = concat(lstm_out[:, t - 1][:, 0 * each_block_size:1 * each_block_size],
                                      torch.zeros(batch_size, (self.block_size - 1) * each_block_size).to(self.device))
                elif fwd_t == (self.block_size - 1):
                    zt_1_tmp = concat(torch.zeros(batch_size, (self.block_size - 2) * each_block_size).to(self.device),
                                      lstm_out[:, t - 1][:,
                                      (fwd_t - 1) * each_block_size:(fwd_t + 1) * each_block_size])
                else:
                    zt_1_tmp = concat(
                        lstm_out[:, t - 1][:, (fwd_t - 1) * each_block_size: (fwd_t + 1) * each_block_size],
                        torch.zeros(batch_size, (self.block_size - 2) * each_block_size).to(self.device))
                '''
                if fwd_t == 0:
                    zt_1_tmp = concat(z_post_sample[:, 0 * each_block_size:1 * each_block_size],
                                      torch.zeros(batch_size, (self.block_size - 1) * each_block_size).to(self.device))
                elif fwd_t == (self.block_size - 1):
                    zt_1_tmp = concat(torch.zeros(batch_size, (self.block_size - 1) * each_block_size).to(self.device),
                                      z_post_sample[:, fwd_t * each_block_size:(fwd_t + 1) * each_block_size])
                else:
                    zt_1_tmp = concat(torch.zeros(batch_size, fwd_t * each_block_size).to(self.device),
                                      z_post_sample[:, fwd_t * each_block_size: (fwd_t + 1) * each_block_size],
                                      torch.zeros(batch_size, (self.block_size -1-fwd_t) * each_block_size).to(self.device))
                '''
                z_fwd_list[fwd_t] = self.z_to_c_fwd_list[fwd_t](zt_1_tmp, z_fwd_list[fwd_t], w=wt[:, fwd_t].view(-1, 1))

            z_fwd_all = torch.stack(z_fwd_list, dim=2).mean(dim=2).view(batch_size, self.hidden_dim)  # .mean(dim=2)
            curr_layer[t-1] = z_fwd_all
            # p(xt|zt)
            # zt_obs = concat(z_fwd_all, z_post_sample)
            zt_obs_list.append(z_post_sample)
            # update weight, w0<...<wd<=1, d means block_size
            # wt = self.z_w_function(concat(z_fwd_all,z_post_sample))
            wt = self.z_w_function(z_fwd_all)
            wt = self.cumsoftmax(wt, self.temperature)

            store_wt.append(wt[0].detach().cpu().numpy())

            z_prior_fwd = self.z_prior_out_list(z_fwd_all)
            z_fwd_latent_mean = z_prior_fwd[:, :self.z_dim]
            z_fwd_latent_lar = z_prior_fwd[:, self.z_dim:]

            # store the prior of ct_i
            z_prior_norm_list.append(Normal(z_fwd_latent_mean, torch.sigmoid(z_fwd_latent_lar)))
            post_z_1 = z_post_sample

        raw_outputs = []
        outputs = []

        prev_layer = torch.stack(curr_layer)
        raw_outputs.append(prev_layer)
        #prev_layer = self.lockdrop(prev_layer, self.dropout)
        outputs.append(prev_layer)

        zt_obs_list = torch.stack(zt_obs_list, dim=1)

        return z_post_norm_list, z_prior_norm_list, zt_obs_list, store_wt, raw_outputs, outputs, lstm_out[:,0:seq_size-1]

    def cumsoftmax(self, x, temp=0.5, dim=-1):
        #if self.training:
        #     x = x + sample_gumbel(x.size())
        x = F.softmax(x, dim=dim)
        # x = torch.log(x)/temp
        # x = torch.exp(x)
        # x = x / torch.sum(x)
        x = torch.cumsum(x, dim=dim)
        return x

    def forward(self, x, temp):
        num_samples = x.shape[0]
        seq_len = x.shape[1]
        conv_x = self.enc_obs(x.view(-1, *x.size()[2:])).view(num_samples, seq_len, -1)
        z_post_norm_list, z_prior_norm_list, z, store_wt, raw_outputs, outputs, lstm_out = self.encode_z(conv_x)
        recon_x = self.dec_obs(z.view(num_samples * seq_len, -1)).view(num_samples, seq_len, *x.size()[2:])
        return z_post_norm_list, z_prior_norm_list, z, recon_x, store_wt, raw_outputs, outputs, lstm_out


def kl_loss_compute(pred1, pred2):
    loss = torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))
    return loss


def loss_fn(dataset, original_seq, recon_seq, z_post_norm, z_prior_norm, raw_outputs, outputs, alpha, beta, eta, kl_weight, lstm_out):
    if dataset == 'lpc':
        obs_cost = F.mse_loss(recon_seq, original_seq, size_average=False)
    elif dataset == 'moving_mnist' or dataset == 'bouncing_balls':
        obs_cost = F.binary_cross_entropy(recon_seq, original_seq, size_average=False)  # binary_cross_entropy
    batch_size = recon_seq.shape[0]
    # compute kl related to states, kl(q(ct|ot,ft)||p(ct|zt-1))

    loss = 0.0
    # Activiation Regularization
    if alpha:
        loss = loss + sum(
            alpha * dropped_rnn_h.pow(2).sum()
            for dropped_rnn_h in outputs[-1:]
        )
    # Temporal Activation Regularization (slowness)
    if beta:
        loss = loss + sum(
            beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).sum()
            for rnn_h in raw_outputs[-1:]
        )

    kl_fwd = (raw_outputs[0].view((lstm_out.shape)) - lstm_out).pow(2).sum() * eta
    kl_fwd += raw_outputs[0].pow(2).sum() *eta
    kl_fwd += lstm_out.pow(2).sum() *eta


    kl_abs_state_list = []
    for t in range(original_seq.shape[1]):
        # kl divergences (sum over dimension)
        kl_abs_state = kl_divergence(z_post_norm[t], z_prior_norm[t])
        kl_abs_state_list.append(kl_abs_state.sum(-1))

    kl_abs_state_list = torch.stack(kl_abs_state_list, dim=1)

    return (obs_cost + kl_weight * (kl_abs_state_list.sum() + kl_fwd) + loss) / batch_size, kl_weight * (kl_abs_state_list.sum()) / batch_size, kl_weight * kl_fwd / batch_size


class Trainer(object):
    def __init__(self, model, device, train, test, epochs, batch_size, learning_rate, nsamples,
                 sample_path, recon_path, checkpoints, log_path, grad_clip, channel, alpha, beta,eta , kl_weight):
        self.train = train
        self.test = test
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.grad_clip = grad_clip
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.samples = nsamples
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.log_path = log_path
        self.epoch_losses = []
        self.channel = channel
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.kl_weight = kl_weight
        self.shape = 32

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.epoch_losses},
            self.checkpoints)

    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def sample_frames(self, epoch, sample):
        with torch.no_grad():
            each_block_size = self.model.hidden_dim // self.model.block_size
            zt_dec = []
            len = sample.shape[0]
            # len = self.samples
            x = self.model.enc_obs(sample.view(-1, *sample.size()[2:])).view(1, sample.shape[1], -1)
            '''
            lstm_out = x.new_zeros(len, self.model.frames, self.model.hidden_dim)
            z_fwd_post = torch.zeros(len, self.model.hidden_dim).to(self.device)
            for i in range(self.model.frames):
                z_fwd_post = self.model.z_lstm(x[:, i], z_fwd_post)
                lstm_out[:, i] = z_fwd_post
            '''
            lstm_out, _ = self.model.z_lstm(x)
            lstm_out, _ = self.model.z_rnn(lstm_out)

            zt_1_post = self.model.z_post_out(lstm_out[:, 0])
            zt_1_mean = zt_1_post[:, :self.model.z_dim]
            zt_1_lar = zt_1_post[:, self.model.z_dim:]

            zt_1 = Normal(zt_1_mean, torch.sigmoid(zt_1_lar)).rsample()
            #zt_1 = [Normal(torch.zeros(self.model.z_dim).to(self.device), torch.ones(self.model.z_dim).to(self.device)).rsample() for i in range(len)]
            #zt_1 = torch.stack(zt_1, dim=0)

            hidden_zt = lstm_out[:, 0]
            #hidden_zt = Normal(torch.zeros(len, self.model.hidden_dim).to(self.device), torch.ones(len, self.model.hidden_dim).to(self.device)).rsample()
            # init wt
            wt = torch.ones(len, self.model.block_size).to(self.device)
            '''
            store_wt.append(wt[0].detach().cpu().numpy())
            '''
            store_wt = []
            z_fwd_list = [torch.zeros(len, self.model.hidden_dim).to(self.device) for i in range(self.model.block_size)]

            zt_dec.append(zt_1)
            for t in range(1, self.model.frames):
                for fwd_t in range(self.model.block_size):
                    # prior over ct of each block, ct_i~p(ct_i|zt-1_i)

                    if fwd_t == 0:
                        zt_1_tmp = concat(hidden_zt[:, 0 * each_block_size:1 * each_block_size],
                                          torch.zeros(len, (self.model.block_size - 1) * each_block_size).to(
                                              self.device))
                    elif fwd_t == (self.model.block_size - 1):
                        zt_1_tmp = concat(
                            torch.zeros(len, (self.model.block_size - 2) * each_block_size).to(self.device),
                            hidden_zt[:, (fwd_t - 1) * each_block_size:(fwd_t + 1) * each_block_size])
                    else:
                        zt_1_tmp = concat(hidden_zt[:, (fwd_t - 1) * each_block_size: (fwd_t + 1) * each_block_size],
                                          torch.zeros(len, (self.model.block_size - 2) * each_block_size).to(
                                              self.device))
                    '''
                    if fwd_t == 0:
                        zt_1_tmp = concat(zt_1[:, 0 * each_block_size:1 * each_block_size],
                                          torch.zeros(len, (self.model.block_size - 1) * each_block_size).to(self.device))
                    elif fwd_t == (self.model.block_size - 1):
                        zt_1_tmp = concat(torch.zeros(len, (self.model.block_size - 1) * each_block_size).to(self.device),
                            zt_1[:, fwd_t * each_block_size:(fwd_t + 1) * each_block_size])
                    else:
                        zt_1_tmp = concat(torch.zeros(len, fwd_t * each_block_size).to(self.device),
                                          zt_1[:, fwd_t * each_block_size: (fwd_t + 1) * each_block_size],
                                          torch.zeros(len, (self.model.block_size - 1 - fwd_t) * each_block_size).to(self.device))
                    '''
                    z_fwd_list[fwd_t] = self.model.z_to_c_fwd_list[fwd_t](zt_1_tmp, z_fwd_list[fwd_t],
                                                                          w=wt[:, fwd_t].view(-1, 1))

                z_fwd_all = torch.stack(z_fwd_list, dim=2).mean(dim=2).view(len, self.model.hidden_dim)  # .mean(dim=2)
                # update weight, w0<...<wd<=1, d means block_size

                z_prior_fwd = self.model.z_prior_out_list(z_fwd_all)
                z_fwd_latent_mean = z_prior_fwd[:, :self.model.z_dim]
                z_fwd_latent_lar = z_prior_fwd[:, self.model.z_dim:]

                # store the prior of ct_i
                zt = Normal(z_fwd_latent_mean, torch.sigmoid(z_fwd_latent_lar)).rsample()
                # zt_obs = concat(z_fwd_all, zt)
                # zt = gumbel_softmax(zt,1.0,self.model.z_dim)
                zt_dec.append(zt)

                # wt = self.model.z_w_function(concat(z_fwd_all, zt))
                wt = self.model.z_w_function(z_fwd_all)
                wt = self.model.cumsoftmax(wt, self.model.temperature)
                store_wt.append(wt[0].detach().cpu().numpy())
                # decode observation
                hidden_zt = z_fwd_all

            zt_dec = torch.stack(zt_dec, dim=1)
            recon_x = self.model.dec_obs(zt_dec.view(len * self.model.frames, -1)).view(len, self.model.frames, -1)
            recon_x = recon_x.view(len * sample.shape[1], self.channel, self.shape, self.shape)
            torchvision.utils.save_image(recon_x, '%s/epoch%d.png' % (self.sample_path, epoch))
            return store_wt

    def recon_frame(self, epoch, original):
        with torch.no_grad():
            _, _, _, recon, store_wt, _, _, _ = self.model(original, 1.0)
            image = torch.cat((original, recon), dim=0)
            print(image.shape)
            image = image.view(2 * original.shape[1], channel, self.shape, self.shape)
            torchvision.utils.save_image(image, '%s/epoch%d.png' % (self.recon_path, epoch))
            return store_wt

    def train_model(self):

        # The number of epochs at which KL loss should be included
        klstart = 40
        # number of epochs over which KL scaling is increased from 0 to 1
        kl_annealtime = 20
        self.model.eval()
        sample = iter(self.test).next().to(self.device)
        print(sample.shape)
        print(sample[0, 0, :, 0, 0].cpu().numpy())
        store_wt = self.sample_frames(0 + 1, sample)
        store_wt1 = self.recon_frame(0 + 1, sample)
        write_log(store_wt, self.log_path)
        write_log('************************', self.log_path)
        write_log(store_wt1, self.log_path)
        self.model.train()
        temp = FLAGS.temperature
        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            kl_loss = []
            kld_fwd_loss = []

            if epoch > klstart:
                self.kl_weight = min(self.kl_weight + (1. / kl_annealtime), 1.)

            write_log("Running Epoch : {}".format(epoch + 1), self.log_path)
            for i, data in enumerate(self.train):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                z_post_norm, z_prior_norm, z, recon_x, _, raw_outputs, outputs, lstm_out = self.model(
                    data, temp)
                loss, kld_z, kld_fwd = loss_fn(self.model.dataset, data, recon_x, z_post_norm, z_prior_norm, raw_outputs, outputs,
                                                       self.alpha, self.beta, self.eta, self.kl_weight, lstm_out)
                write_log('mse loss is %f, kl loss is %f, kl fwd loss is %f' % (loss, kld_z, kld_fwd),
                    self.log_path)
                print('index is %d, mse loss is %f, kl loss is %f, kl fwd loss is %f' % (
                i, loss, kld_z, kld_fwd))
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                loss.backward()
                self.optimizer.step()
                kl_loss.append(kld_z.item())

                kld_fwd_loss.append(kld_fwd.item())
                losses.append(loss.item())
            meanloss = np.mean(losses)
            klloss = np.mean(kl_loss)
            kldfwdloss = np.mean(kld_fwd_loss)
            self.epoch_losses.append((meanloss, klloss))
            write_log(
                "Epoch {} : Average Loss: {}, KL loss :{}, kl fwd loss :{}".format(epoch + 1, meanloss
                                                                                                 , klloss,
                                                                                                 kldfwdloss),
                self.log_path)
            # self.save_checkpoint(epoch)
            self.model.eval()
            sample = iter(self.test).next().to(self.device)
            store_wt = self.sample_frames(epoch + 1, sample)
            store_wt1 = self.recon_frame(epoch + 1, sample)
            write_log(store_wt, self.log_path)
            write_log('************************', self.log_path)
            write_log(store_wt1, self.log_path)
            self.model.train()
        print("Training is complete")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="hybrid_vae")
    parser.add_argument('--seed', type=int, default=111)
    # method
    parser.add_argument('--method', type=str, default='Hybrid_1_1')
    # dataset
    parser.add_argument('--dset_name', type=str, default='bouncing_balls')  # moving_mnist, lpc, bouncing_balls
    # state size
    parser.add_argument('--z-dim', type=int, default=72)  # 72 144
    parser.add_argument('--hidden-dim', type=int, default=216)  # 216 252
    parser.add_argument('--conv-dim', type=int, default=256)  # 256 512
    parser.add_argument('--block_size', type=int, default=3)  # 3  4
    # data size
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--frame-size', type=int, default=20)
    parser.add_argument('--nsamples', type=int, default=2)

    # optimization
    parser.add_argument('--learn-rate', type=float, default=0.0005)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=1,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=1.0)

    FLAGS = parser.parse_args()
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    device = torch.device('cuda:%d' % (FLAGS.gpu_id) if torch.cuda.is_available() else 'cpu')

    if FLAGS.dset_name == 'lpc':
        sprite = Sprites('./dataset/lpc-dataset/train/', 6687)
        sprite_test = Sprites('./dataset/lpc-dataset/test/', 873)
        train_loader = torch.utils.data.DataLoader(sprite, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(sprite_test, batch_size=1, shuffle=FLAGS, num_workers=4)
        channel = 3
        shape = 64
    elif FLAGS.dset_name == 'moving_mnist':
        FLAGS.dset_path = os.path.join('./datasets', FLAGS.dset_name)
        train_loader, test_loader = data.get_data_loader(FLAGS, True)
        channel = 1
        shape = 64
    elif FLAGS.dset_name == 'bouncing_balls':
        sprite = bouncing_balls('./bouncing_ball/dataset/', 6000)
        sprite_test = bouncing_balls('./bouncing_ball/dataset/', 300)
        train_loader = torch.utils.data.DataLoader(sprite, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(sprite_test, batch_size=1, shuffle=True, num_workers=4)
        channel = 3
        shape = 32

    vae = FullQDisentangledVAE(temperature=FLAGS.temperature, frames=FLAGS.frame_size, z_dim=FLAGS.z_dim,
                               hidden_dim=FLAGS.hidden_dim,
                               conv_dim=FLAGS.conv_dim, block_size=FLAGS.block_size, channel=channel,
                               dataset=FLAGS.dset_name, device=device, shape=shape)

    starttime = datetime.datetime.now()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    time_dir = now.strftime('%Y_%m_%d_%H_%M_%S')
    base_path = './%s/%s/%s' % (FLAGS.dset_name, FLAGS.method, time_dir)
    model_path = '%s/model' % (base_path)
    log_recon = '%s/recon' % (base_path)
    log_sample = '%s/sample' % (base_path)
    log_path = '%s/log_info.txt' % (base_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_recon):
        os.makedirs(log_recon)
    if not os.path.exists(log_sample):
        os.makedirs(log_sample)

    write_log(vae, log_path)
    write_log(vae.z_to_c_fwd_list, log_path)
    write_log(FLAGS, log_path)

    trainer = Trainer(vae, device, train_loader, test_loader, epochs=FLAGS.max_epochs, batch_size=FLAGS.batch_size,
                      learning_rate=FLAGS.learn_rate,
                      checkpoints='%s/%s-disentangled-vae.model' % (model_path, FLAGS.method), nsamples=FLAGS.nsamples,
                      sample_path=log_sample,
                      recon_path=log_recon, log_path=log_path, grad_clip=FLAGS.grad_clip, channel=channel,
                      alpha=FLAGS.alpha, beta=FLAGS.beta,eta=FLAGS.eta, kl_weight=FLAGS.kl_weight)
    # trainer.load_checkpoint()
    trainer.train_model()
    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    second = (seconds % 3600) % 60
    print((endtime - starttime))
    timeStr = "running time: " + str(hours) + 'hours' + str(minutes) + 'minutes' + str(second) + "second"
    write_log(timeStr, log_path)
