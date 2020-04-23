import os
import torchvision
import torch.utils.data
import torch.nn.init
import torch.optim as optim
from GRU_cell import GRUCell
from modules import *
import numpy as np
import datetime
import dateutil.tz
import argparse
import data

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


class FullQDisentangledVAE(nn.Module):
    def __init__(self, frames, z_dim, conv_dim, hidden_dim, device):
        super(FullQDisentangledVAE, self).__init__()
        self.z_dim = z_dim
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.block_size = 3
        self.device = device

        self.z_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        self.z_post_out = nn.Linear(self.hidden_dim, self.z_dim * 2)

        self.z_prior_out = nn.Linear(self.hidden_dim, self.z_dim * 2)

        self.z_to_c_fwd_list = [
            GRUCell(input_size=self.z_dim, hidden_size=self.hidden_dim // self.block_size).to(device)
            for i in range(self.block_size)]

        self.z_w_function = nn.Linear(self.hidden_dim, self.block_size)
        # observation encoder / decoder
        self.enc_obs = Encoder(feat_size=self.hidden_dim, output_size=self.conv_dim)
        self.dec_obs = Decoder(input_size=self.z_dim,
                               feat_size=self.hidden_dim)

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def encode_z(self, x):
        lstm_out, _ = self.z_lstm(x)
        lstm_out, _ = self.z_rnn(lstm_out)

        batch_size = lstm_out.shape[0]
        seq_size = lstm_out.shape[1]
        each_block_size = self.z_dim // self.block_size

        z_post_mean_list = []
        z_post_lar_list = []
        z_prior_mean_list = []
        z_prior_lar_list = []
        zt_obs_list = []

        zt_1_post = self.z_post_out(lstm_out[:, 0])
        zt_1_mean = zt_1_post[:, :self.z_dim]
        zt_1_lar = zt_1_post[:, self.z_dim:]

        post_z_1 = self.reparameterize(zt_1_mean, zt_1_lar, self.training)
        zt_obs_list.append(post_z_1)

        # init wt
        wt = torch.ones(batch_size, self.block_size).to(device)
        z_fwd_list = [torch.zeros(batch_size, self.hidden_dim//self.block_size).to(device) for i in range(self.block_size)]
        for t in range(1, seq_size):

            z_post_out = self.z_post_out(lstm_out[:, t])
            zt_post_mean = z_post_out[:, :self.z_dim]
            zt_post_lar = z_post_out[:, self.z_dim:]

            z_post_mean_list.append(zt_post_mean)
            z_post_lar_list.append(zt_post_lar)
            z_post_sample = self.reparameterize(zt_post_mean, zt_post_lar, self.training)
            # p(xt|zt)
            zt_obs_list.append(z_post_sample)

            for fwd_t in range(self.block_size):
                # prior over ct of each block, ct_i~p(ct_i|zt-1_i)
                if fwd_t==0:
                    zt_1_tmp = concat(post_z_1[:, fwd_t * each_block_size:(fwd_t + 2) * each_block_size], torch.zeros(batch_size, self.z_dim//self.block_size).to(device))
                if fwd_t==1:
                    zt_1_tmp = post_z_1
                if fwd_t==2:
                    zt_1_tmp = concat(torch.zeros(batch_size, self.z_dim//self.block_size).to(device),post_z_1[:, (fwd_t-1) * each_block_size:(fwd_t + 1) * each_block_size])

                z_fwd_list[fwd_t] = self.z_to_c_fwd_list[fwd_t](zt_1_tmp, z_fwd_list[fwd_t],w=wt[:,fwd_t].view(-1,1))

            z_fwd_all = torch.stack(z_fwd_list, dim=2).view(batch_size, self.hidden_dim)
            # update weight, w0<...<wd<=1, d means block_size
            #wt = self.z_w_function(z_fwd_all)
            #wt = cumsoftmax(wt)

            z_prior_fwd = self.z_prior_out(z_fwd_all)

            z_fwd_latent_mean = z_prior_fwd[:, :self.z_dim]
            z_fwd_latent_lar = z_prior_fwd[:, self.z_dim:]

            # store the prior of ct_i
            z_prior_mean_list.append(z_fwd_latent_mean)
            z_prior_lar_list.append(z_fwd_latent_lar)

            post_z_1 = z_post_sample

        zt_obs_list = torch.stack(zt_obs_list, dim=1)
        z_post_mean_list = torch.stack(z_post_mean_list, dim=1)
        z_post_lar_list = torch.stack(z_post_lar_list, dim=1)
        z_prior_mean_list = torch.stack(z_prior_mean_list, dim=1)
        z_prior_lar_list = torch.stack(z_prior_lar_list, dim=1)

        return zt_1_mean, zt_1_lar, z_post_mean_list, z_post_lar_list, z_prior_mean_list, z_prior_lar_list, zt_obs_list

    def forward(self, x):
        num_samples = x.shape[0]
        seq_len = x.shape[1]
        conv_x = self.enc_obs(x.view(-1, *x.size()[2:])).view(num_samples, seq_len, -1)
        zt_1_mean, zt_1_lar, post_zt_mean, post_zt_lar, prior_zt_mean, prior_zt_lar, z = self.encode_z(conv_x)
        recon_x = self.dec_obs(z.view(num_samples * seq_len, -1)).view(num_samples, seq_len, *x.size()[2:])
        return zt_1_mean, zt_1_lar, post_zt_mean, post_zt_lar, prior_zt_mean, prior_zt_lar, z, recon_x

def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)

def loss_fn(original_seq, recon_seq, zt_1_mean, zt_1_lar,z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):

    obs_cost = F.mse_loss(recon_seq, original_seq,reduction='sum')   #binary_cross_entropy
    batch_size = recon_seq.shape[0]

    # compute kl related to states, kl(q(ct|ot,ft)||p(ct|zt-1))
    kld_z0 = -0.5 * torch.sum(1 + zt_1_lar - torch.pow(zt_1_mean, 2) - torch.exp(zt_1_lar))
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * torch.sum(
        z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
    return (obs_cost + kld_z + kld_z0)/batch_size , (kld_z + kld_z0)/batch_size


class Trainer(object):
    def __init__(self, model, device, train, test, epochs, batch_size, learning_rate, nsamples,
                 sample_path, recon_path, checkpoints, log_path, grad_clip):
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
            each_block_size = self.model.z_dim // self.model.block_size
            zt_dec = []
            len = sample.shape[0]
            #len = self.samples
            x = self.model.enc_obs(sample.view(-1, *sample.size()[2:])).view(1, 8, -1)
            lstm_out, _ = self.model.z_lstm(x)
            lstm_out, _ = self.model.z_rnn(lstm_out)

            zt_1_post = self.model.z_post_out(lstm_out[:, 0])
            zt_1_mean = zt_1_post[:, :self.model.z_dim]
            zt_1_lar = zt_1_post[:, self.model.z_dim:]

            zt_1 = self.model.reparameterize(zt_1_mean, zt_1_lar, self.model.training)
            #zt_1 = [Normal(torch.zeros(self.model.z_dim).to(self.device), torch.ones(self.model.z_dim).to(self.device)).rsample() for i in range(len)]
            #zt_1 = torch.stack(zt_1, dim=0)

            zt_dec.append(zt_1)
            # init wt
            wt = torch.ones(len, self.model.block_size).to(device)
            z_fwd_list = [torch.zeros(len, self.model.hidden_dim//self.model.block_size).to(device) for i in range(self.model.block_size)]
            for t in range(1, self.model.frames):
                for fwd_t in range(self.model.block_size):
                    # prior over ct of each block, ct_i~p(ct_i|zt-1_i)
                    if fwd_t == 0:
                        zt_1_tmp = concat(zt_1[:, fwd_t * each_block_size:(fwd_t + 2) * each_block_size],
                                          torch.zeros(len, self.model.z_dim // self.model.block_size).to(device))
                    if fwd_t == 1:
                        zt_1_tmp = zt_1
                    if fwd_t == 2:
                        zt_1_tmp = concat(torch.zeros(len, self.model.z_dim//self.model.block_size).to(device),
                                          zt_1[:, (fwd_t - 1) * each_block_size:(fwd_t + 1) * each_block_size])

                    z_fwd_list[fwd_t] = self.model.z_to_c_fwd_list[fwd_t](zt_1_tmp, z_fwd_list[fwd_t], wt[:, fwd_t].view(-1,1))

                z_fwd_all = torch.stack(z_fwd_list, dim=2).view(len, self.model.hidden_dim)
                # update weight, w0<...<wd<=1, d means block_size
                #wt = self.model.z_w_function(z_fwd_all)
                #wt = cumsoftmax(wt)

                z_prior_fwd = self.model.z_prior_out(z_fwd_all)
                z_fwd_latent_mean = z_prior_fwd[:, :self.model.z_dim]
                z_fwd_latent_lar = z_prior_fwd[:, self.model.z_dim:]

                # store the prior of ct_i
                zt = self.model.reparameterize(z_fwd_latent_mean, z_fwd_latent_lar, self.model.training)
                zt_dec.append(zt)
                # decode observation
                zt_1 = zt

            zt_dec = torch.stack(zt_dec, dim=1)
            recon_x = self.model.dec_obs(zt_dec.view(len * self.model.frames, -1)).view(len, self.model.frames, -1)
            recon_x = recon_x.view(len * 8, 1, 64, 64)
            torchvision.utils.save_image(recon_x, '%s/epoch%d.png' % (self.sample_path, epoch))

    def recon_frame(self, epoch, original):
        with torch.no_grad():
            _, _, _, _, _, _,_, recon = self.model(original)
            image = torch.cat((original, recon), dim=0)
            print(image.shape)
            image = image.view(16, 1, 64, 64)
            torchvision.utils.save_image(image, '%s/epoch%d.png' % (self.recon_path, epoch))

    def train_model(self):
        self.model.train()
        sample = iter(self.test).next().to(self.device)
        self.sample_frames(0 + 1, sample)
        self.recon_frame(0 + 1, sample)
        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            kl_loss = []
            write_log("Running Epoch : {}".format(epoch + 1), self.log_path)
            for i, data in enumerate(self.train, 1):
                data = data.to(device)
                self.optimizer.zero_grad()
                zt_1_mean, zt_1_lar, post_zt_mean, post_zt_lar, prior_zt_mean, prior_zt_lar, z, recon_x = self.model(data)
                loss, kl = loss_fn(data, recon_x, zt_1_mean, zt_1_lar, post_zt_mean, post_zt_lar, prior_zt_mean, prior_zt_lar)
                print('mse loss is %f, kl loss is %f'%(loss, kl))
                if self.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                loss.backward()
                self.optimizer.step()
                kl_loss.append(kl.item())
                losses.append(loss.item())
            meanloss = np.mean(losses)
            klloss = np.mean(kl_loss)
            self.epoch_losses.append((meanloss,klloss))
            write_log("Epoch {} : Average Loss: {}, KL loss :{}".format(epoch + 1, meanloss, klloss), self.log_path)
            self.save_checkpoint(epoch)
            self.model.eval()
            sample = iter(self.test).next().to(self.device)
            self.sample_frames(epoch + 1, sample)
            self.recon_frame(epoch + 1, sample)
            self.model.train()
        print("Training is complete")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="hybrid_vae")
    parser.add_argument('--seed', type=int, default=111)
    # method
    parser.add_argument('--method', type=str, default='Hybrid')
    # dataset
    parser.add_argument('--dset_name', type=str, default='moving_mnist')
    # state size
    parser.add_argument('--z-dim', type=int, default=144)  # 72 144
    parser.add_argument('--hidden-dim', type=int, default=252) #  216 252
    parser.add_argument('--conv-dim', type=int, default=256)  # 256 512
    # data size
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--frame-size', type=int, default=8)
    parser.add_argument('--nsamples', type=int, default=2)

    # optimization
    parser.add_argument('--learn-rate', type=float, default=0.0005)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--max-epochs', type=int, default=300)
    parser.add_argument('--gpu_id', type=int, default=1)

    FLAGS = parser.parse_args()
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    device = torch.device('cuda:%d' % (FLAGS.gpu_id) if torch.cuda.is_available() else 'cpu')

    vae = FullQDisentangledVAE(frames=FLAGS.frame_size, z_dim=FLAGS.z_dim, hidden_dim=FLAGS.hidden_dim, conv_dim=FLAGS.conv_dim, device=device)

    FLAGS.dset_path = os.path.join('./datasets', FLAGS.dset_name)
    train_loader, test_loader = data.get_data_loader(FLAGS, True)

    starttime = datetime.datetime.now()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    time_dir = now.strftime('%Y_%m_%d_%H_%M_%S')
    base_path = './%s/%s'%(FLAGS.method, time_dir)
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
    write_log(FLAGS, log_path)

    trainer = Trainer(vae, device, train_loader, test_loader, epochs=FLAGS.max_epochs, batch_size=FLAGS.batch_size,
                      learning_rate=FLAGS.learn_rate,
                      checkpoints='%s/%s-disentangled-vae.model' % (model_path, FLAGS.method), nsamples=FLAGS.nsamples,
                      sample_path=log_sample,
                      recon_path=log_recon, log_path=log_path, grad_clip=FLAGS.grad_clip)
    #trainer.load_checkpoint()
    trainer.train_model()
    endtime = datetime.datetime.now()
    seconds = (endtime - starttime).seconds
    hours = seconds // 3600
    minutes = (seconds % 3600)//60
    second = (seconds % 3600)%60
    print((endtime - starttime))
    timeStr = "running time: " + str(hours)+ 'hours'+ str(minutes) + 'minutes' + str(second) + "second"
    write_log(timeStr, log_path)
