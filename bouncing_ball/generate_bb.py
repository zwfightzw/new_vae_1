from .bouncing_balls import bounce_vec

import numpy as np
import cv2
import torch
from PIL import Image

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
    seq_len = 2 * seq_length
    dat = np.zeros((batch_size, seq_len, shape, shape, 3), dtype=np.float32)
    for i in range(batch_size):
        dat[i, :, :, :, :] = bounce_vec(64, num_balls, seq_len)
    index = [i*2 for i in range(seq_length)]
    return dat[0,index,:,:,:].squeeze().transpose((0,3,1,2))


'''
batch_size=2
shape = [64, 64]
num_balls=2
seq_len = 8
dat = generate_bouncing_ball_sample(batch_size, seq_len, shape[0], num_balls)

fps = 1

for i in range(batch_size):
    for j in range(seq_len):
        #print(dat[0][0])
        cv2.imwrite('tmp_%d.jpg'%(j),dat[i][j]*255)
        img = cv2.imread('tmp.jpg')

'''
