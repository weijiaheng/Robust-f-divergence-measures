import numpy as np
import csv
import torch.nn.parallel
import torch.optim
import torch.utils.data
import pickle
from tqdm import tqdm
import torch.nn as nn
from model import *
from noise_data_cifar_100 import *
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import argparse
torch.autograd.set_detect_anomaly(True)
import math
num_classes = 100
num_epochs = 240

CUDA = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


opt = parser.parse_args()

# Stable CE
class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )

        
criterion = CrossEntropyLossStable()
criterion.cuda()

# Losses

def train(train_loader, model, optimizer, epoch):

    model.train()
    for i, (idx, input, target) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            continue
        warmup_epoch = args.warmup
        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())
        output = model(input)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.cuda()
        loss.backward()
        optimizer.step()


def test(model, test_loader, is_test = True):
    model.eval()
    correct = 0
    total = 0

    for i, (idx, input, target) in enumerate(test_loader):
        input = torch.Tensor(input).cuda()
        if is_test == False:
            target = torch.autograd.Variable(target).cuda()
        else:
            target = torch.autograd.Variable(target[0]).cuda()
        total += target.size(0)
        output = model(input)
        _, predicted = torch.max(output.detach(), 1)
        correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / total

    return accuracy


def main(writer):
    model_ce = resnet_cifar18_pre(num_classes=100).cuda()
    best_ce_acc = 0
    val_acc_noisy_result = []
    train_acc_result = []
    test_acc_result = []
    for epoch in range(num_epochs):
        print("epoch=", epoch,'r=', args.r)
        learning_rate = 0.1
        if epoch >=60:
            learning_rate = 0.01
        if epoch >= 120:
            learning_rate = 0.001
        if epoch >= 180:
            learning_rate = 0.0001
        optimizer_ce = torch.optim.SGD(model_ce.parameters(), momentum=0.9, weight_decay=5e-4, lr=learning_rate)
        train(train_loader=train_loader_noisy, model=model_ce, optimizer=optimizer_ce, epoch=epoch)
        print("validating model_ce...")
        train_acc = test(model=model_ce, test_loader=train_loader_noisy, is_test = False)
        train_acc_result.append(train_acc)
        print('train_acc=', train_acc)
        valid_acc = test(model=model_ce, test_loader=valid_loader_noisy, is_test = False)
        val_acc_noisy_result.append(valid_acc)
        print('valid_acc_noise=', valid_acc)
        test_acc = test(model=model_ce, test_loader=test_loader_, is_test = False)
        test_acc_result.append(test_acc)
        print('test_acc=', test_acc)
        if valid_acc >= best_ce_acc:
            best_ce_acc = valid_acc
            torch.save(model_ce, './trained_models/ce' + str(args.r) + '_' + str(args.s) + '_')
            print("saved, valid acc increases.")
        writer.writerow([epoch, train_acc, valid_acc, test_acc])
    



def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_, is_test = False)
    print('test_acc=', test_acc)



if __name__ == '__main__':
   
    print("Begin:")
    writer1 = csv.writer(open(f'CIFAR100_CE_result_{r}.csv','w'))
    writer1.writerow(['Epoch', 'Training Acc', f'Val_Noisy_Acc', 'Test_ACC'])
    main(writer1)
    evaluate('./trained_models/ce' + str(args.r) + '_' + str(args.s) + '_')
    print("Traning finished")
