# dataset - preparing data and adding label noise
from __future__ import print_function
import os
import os.path
import codecs
import numpy as np
import sys
import argparse
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
from PIL import Image
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

# r: noise amount s: random seed
parser = argparse.ArgumentParser()
parser.add_argument('--r', type=float, required=True, help='category of noise label')
parser.add_argument('--s', type=int, default=10086, help='random seed')
parser.add_argument('--divergence', required=True, help='f-divergence category')
parser.add_argument('--warmup', type=int, required=True, help='number of warm-up epochs')
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--bias', type=str, default=False, help='assume do not use bias correction')
parser.add_argument('--root', type=str, default="./datasets/", help='Path for loading the dataset')
args = parser.parse_args()
CUDA = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

# data root:
root = args.root
r = args.r
batch_size = args.batchsize
class Fashion_(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    
    def __init__(self, root, train=True, valid = False, test = False, noisy=False,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.valid = valid
        self.test = test
        self.noisy = noisy

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                           ' You can use download=True to download it')

        train_data, train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))

        test_data, test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))

        self.train_data = train_data[0:50000]
        self.valid_data = train_data[50000:60000]
        self.test_data = test_data
        
        self.train_label = train_labels[0:50000]
        self.valid_label = train_labels[50000:60000]
        self.test_label = test_labels

        if noisy == True:
            if self.valid:
                t_ = self._load_noise_label(True)[50000:60000]
                self.valid_label = t_.tolist()
            elif self.train:
                t_ = self._load_noise_label(True)[:50000]
                self.train_label = t_.tolist()
            

    def _load_noise_label(self, is_train):
        '''
        I adopte .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        dataset_path = "./noise_label"
        
        key = 'noise_label_test'
        fname = "test_label.pt"
        if is_train:
            key = 'noise_label_train'
            num = int(r*10)
            fname = f"Fashion_noise_r0_{num}.pt"

        fpath = os.path.join(dataset_path, fname)
        if is_train:
            noise_label_torch = torch.load(fpath)
            noise_label_torch = noise_label_torch[key]
        else:
            noise_label_torch = torch.load(fpath)
        return noise_label_torch


    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        elif self.valid:
            img, target = self.valid_data[index], self.valid_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return index, img, target


    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.valid:
            return len(self.valid_data)
        else:
            return len(self.test_data)
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')



def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

transform_test = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])


def peer_data_train(batch_size, img_size=(32, 32)):
    peer_dataset = Fashion_(root=root, train=True, valid=False, test=False, noisy=True, transform=transform_train)
   
    peer_dataloader = torch.utils.data.DataLoader(
        peer_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
     
    return peer_dataloader


def peer_data_val(batch_size, img_size=(32, 32)):
    peer_dataset = Fashion_(root=root, train=False, valid=True, test=False, noisy=True, transform=transform_train)

    peer_dataloader = torch.utils.data.DataLoader(
        peer_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
     
    return peer_dataloader
    
def peer_data_test(batch_size, img_size=(32, 32)):
    peer_dataset = Fashion_(root=root, train=False, valid=False, test=True, noisy=False, transform=transform_train)

    peer_dataloader = torch.utils.data.DataLoader(
        peer_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
     
    return peer_dataloader

train_dataset_noisy = Fashion_(root=root, train=True, valid=False, test=False, noisy=True, transform=transform_train)
        
train_loader_noisy = torch.utils.data.DataLoader(dataset=train_dataset_noisy, batch_size=batch_size, shuffle=True)

valid_dataset_noisy = Fashion_(root=root, train=False, valid=True, test=False, noisy=True, transform=transform_train)
        
valid_loader_noisy = torch.utils.data.DataLoader(dataset = valid_dataset_noisy, batch_size = batch_size, shuffle = False)

test_dataset = Fashion_(root=root, train=False, valid=False, test=True, noisy=False, transform=transform_test)

test_loader_ = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)

        return out
