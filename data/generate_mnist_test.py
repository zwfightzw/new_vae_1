from PIL import Image
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import data.video_transforms as vtransforms
from moving_mnist import MovingMNIST
from bouncing_balls import BouncingBalls
import numpy as np

def get_data_loader(dset_path):
  transform = transforms.Compose([vtransforms.ToTensor()])
  dset = MovingMNIST(dset_path, True, 10,10, [1], transform)
  return dset


dset_path = os.path.join('../datasets', 'moving_mnist')

test_loader = get_data_loader(dset_path)
count = 0

for step, test_obs_list in enumerate(test_loader):
  if count == 0:
    test_store = test_obs_list
  if count ==500: break
  count += 1
  test_store = torch.cat((test_store,test_obs_list),dim=0)

test_store = test_store.reshape(-1,20,1,64,64).data.numpy()
np.save('../datasets/mnist_test_seq', test_store)
print('success')
