# dataset - preparing data and adding label noise
from __future__ import print_function
import torch
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import argparse

# r: noise amount s: random seed
parser = argparse.ArgumentParser()
parser.add_argument('--r', type=float, required=True, help='category of noise label')
parser.add_argument('--s', type=int, default=10086, help='random seed')
parser.add_argument('--divergence', required=True, help='f-divergence category')
parser.add_argument('--warmup', type=int, required=True, help='number of warm-up epochs')
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--root', type=str, default="./datasets", help='Path for loading CIFAR100 dataset')
args = parser.parse_args()

CUDA = True if torch.cuda.is_available() else False


Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

# data root:
root = args.root
r = args.r
batch_size = args.batchsize


class CIFAR100_(data.Dataset):

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
        ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    
    def __init__(self, root, noise_indices = [0,1,2,3,4,5], train=True, valid = False, test = False, noisy=False, transform=None, target_transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.valid = valid
        self.test = test
        self.noisy = noisy

        if download:
            self.download()

        if self.train:
            downloaded_list = self.train_list
        if self.valid:
            downloaded_list = self.train_list
        if self.test:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        if noisy == True:
            if self.valid:
                t_ = self._load_noise_label(True)[40000:]
                self.data = self.data[40000:]
            elif self.train:
                t_ = self._load_noise_label(True)[:40000]
                self.data = self.data[:40000]
            else:
                t_ = self._load_noise_label(False)
            self.targets = t_.tolist()
            
        else:
            if self.valid:
                self.data = self.data[40000:]
                self.targets = self.targets[40000:]
            elif self.train:
                self.data = self.data[:40000]
                self.targets = self.targets[:40000]
            else:
                self.data = self.data
                self.targets = self.targets
        self._load_meta()

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
            fname = f"CIFAR100_noise_r0_{num}.pt"

        fpath = os.path.join(dataset_path, fname)
        if is_train:
            noise_label_torch = torch.load(fpath)
            noise_label_torch = noise_label_torch[key]
        else:
            noise_label_torch = torch.load(fpath)
        return noise_label_torch

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = index

        return idx, img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)


        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])


def peer_data(batch_size, img_size=(32, 32)):
#    os.makedirs("data/cifar100_peer", exist_ok=True)
    peer_dataset = CIFAR100_(root=root, train=True, valid=False, test=False, noisy=True, transform=transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ]))
   
    peer_dataloader = torch.utils.data.DataLoader(
        peer_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
     
    return peer_dataloader


def peer_data_val(batch_size, img_size=(32, 32)):
#    os.makedirs("data/cifar100_peer_val", exist_ok=True)
    peer_dataset = CIFAR100_(root=root, train=False, valid=True, test=False, noisy=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    peer_dataloader = torch.utils.data.DataLoader(
        peer_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
     
    return peer_dataloader

train_dataset_noisy = CIFAR100_(root=root, train=True, valid=False, test=False, noisy=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]))
        
train_loader_noisy = torch.utils.data.DataLoader(dataset=train_dataset_noisy, batch_size=batch_size, shuffle=True, num_workers=0)

train_loader_noisy_unshuffle = torch.utils.data.DataLoader(dataset = train_dataset_noisy, batch_size=batch_size, shuffle =False)

valid_dataset_noisy = CIFAR100_(root=root, train=False, valid=True, test=False, noisy=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

valid_dataset_clean = CIFAR100_(root=root, train=False, valid=True, test=False, noisy=False, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize,]))
        
valid_loader_noisy = torch.utils.data.DataLoader(dataset = valid_dataset_noisy, batch_size = batch_size, shuffle = False, num_workers=0)

valid_loader_clean = torch.utils.data.DataLoader(dataset = valid_dataset_clean, batch_size = batch_size, shuffle = False,  num_workers=0)

test_dataset = CIFAR100_(root=root, train=False, valid=False, test=True, noisy=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
test_loader_ = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers=0)
