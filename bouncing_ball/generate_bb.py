from bouncing_balls import bounce_vec


import numpy as np
import torch
import torchvision

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
    seq_len = 2 * seq_length
    dat = np.zeros((batch_size, seq_len, shape, shape, 3),dtype=np.float32)
    for i in range(batch_size):
        dat[i, :, :, :, :] = bounce_vec(shape, num_balls, seq_len)
    index = [i*2 for i in range(seq_length)]
    return dat[0,index,:,:,:].squeeze().transpose((0,3,1,2))

if __name__ == '__main__':


    batch_size=2
    shape = [64, 64]
    num_balls=2
    seq_len = 200
    dat = generate_bouncing_ball_sample(batch_size, seq_len, shape[0], num_balls)

    fps = 1
    dat = torch.from_numpy(dat)
    recon_x = dat.view((seq_len, 3, shape[0], shape[0]))
    torchvision.utils.save_image(recon_x, 'sample.png')

