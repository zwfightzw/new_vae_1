import os
import numpy as np
import torch.utils.data as data
import cv2
from bouncing_ball.generate_bb import generate_bouncing_ball_sample

class BouncingBalls(data.Dataset):
  '''
  Bouncing balls dataset.
  '''
  def __init__(self, root, n_frames_total,
               transform=None):
    super(BouncingBalls, self).__init__()
    self.n_frames = n_frames_total
    self.transform = transform

    self.shape = (64, 64)   # H,W
    self.num_balls = 2

  def __getitem__(self, idx):
    # traj sizeL (n_frames, n_balls, 4)
    single_sequence = generate_bouncing_ball_sample(1, self.n_frames, self.shape[0], self.num_balls)
    return single_sequence

  def __len__(self):
    return 10000
