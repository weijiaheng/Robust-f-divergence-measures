 # dataset - preparing data and adding label noise
from __future__ import print_function
import torch
import time
from PIL import Image
import os
import os.path
import random
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, RandomSampler,SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
import argparse

# r: noise amount s: random seed
parser = argparse.ArgumentParser()
parser.add_argument('--r', type=float)
parser.add_argument('--s', type=int, default=10086, help='random seed')
parser.add_argument('--divergence', required=True, help='f-divergence category')
parser.add_argument('--warmup', type=int, required=True, help='number of warm-up epochs')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=56)
parser.add_argument('--root', type=str, default="./datasets/", help='Path for loading the dataset')
args = parser.parse_args()
CUDA = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

# data root:
root = args.root
r = args.r
batch_size = args.batchsize


def my_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return None, None, None, None
    return default_collate(batch)


class ClothFolder(ImageFolder):

    def __init__(self, root, transform):
        super(ClothFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        self.img_num = len(self.samples)


    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        except:
                return None,-1

prior = np.array([4290, 4549, 2113, 4415, 3027, 3844, 4236, 2393, 935, 4184, 3970, 4028, 3572, 4444])/50000
labels = [i for i in range(14)]

with open("./C1M_selected_idx_balance.pkl", "rb") as f:
    subsetIdx = pickle.load(f)

print('Preparing train, val, test dataloader ...')

# Modify following paths accordingly
train_folder = "/clothing1m/noisy_train"
val_folder   = "/clothing1m/clean_val"
test_folder  = "/clothing1m/clean_test"

train_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
        ])
test_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
        ])
        


train_dataset_noisy = ClothFolder(root=train_folder, transform=train_trans)
train_sampler_1 = SubsetRandomSampler(subsetIdx)
train_sampler_2 = SubsetRandomSampler(subsetIdx)

        
train_loader_noisy = torch.utils.data.DataLoader(dataset=train_dataset_noisy, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, sampler=train_sampler_1, num_workers=8)

train_peer_loader = torch.utils.data.DataLoader(dataset=train_dataset_noisy, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, sampler=train_sampler_2, num_workers=8)

valid_dataset_noisy = ClothFolder(root=val_folder, transform=test_trans)


valid_loader_noisy = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = int(batch_size/4), shuffle = False, collate_fn=my_collate_fn, num_workers = 14)

val_peer_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = int(batch_size/4), shuffle = True, collate_fn=my_collate_fn, num_workers = 14)


test_dataset = ClothFolder(root=test_folder, transform=test_trans)
test_loader_ = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 16, shuffle = False, collate_fn=my_collate_fn, num_workers = 8)


