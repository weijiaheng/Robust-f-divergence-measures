import numpy as np
import time
import csv
import torch.nn.parallel
import torch.optim
import torch.utils.data
import pickle
from tqdm import tqdm
import torch.nn as nn
from model import *
from dataloader import *
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import argparse
torch.autograd.set_detect_anomaly(True)
import math
import torchvision.models as models
num_classes = 14
num_epochs = 40


opt = parser.parse_args()
torch.cuda.set_device(opt.device)

# Stable version of CE Loss
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

# Divergence functions:
div = opt.divergence

if div == 'KL':
    def activation(x): return -torch.mean(x)
    
    def conjugate(x): return -torch.mean(torch.exp(x - 1.))


elif div == 'Jeffrey':
    def activation(x): return -torch.mean(x)
    
    def conjugate(x): return -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)
    
elif div == 'Pearson':
    def activation(x): return -torch.mean(x)
    def conjugate(x): return -torch.mean(torch.mul(x, x) / 4. + x)

elif div == 'Neyman':
    def activation(x): return -torch.mean(1. - torch.exp(x))

    def conjugate(x): return -torch.mean(2. - 2. * torch.sqrt(1. - x))

elif div == 'Jenson-Shannon':
    def activation(x): return -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))

    def conjugate(x): return -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))

elif div == 'Total-Variation':
    def activation(x): return -torch.mean(torch.tanh(x) / 2.)

    def conjugate(x): return -torch.mean(torch.tanh(x) / 2.)
        
else:
    raise NotImplementedError("[-] Not Implemented f-divergence %s" % div)



# Stable PROB: returns the negative predicted probability of an image given a reference label
class ProbLossStable(nn.Module):
    def __init__(self, reduction='none', eps=1e-5):
        super(ProbLossStable, self).__init__()
        self._name = "Prob Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction='none')

    def forward(self, outputs, labels):
        return self._nllloss( self._softmax(outputs), labels )
        
criterion_prob = ProbLossStable().cuda()


# Training
def train(train_loader, peer_loader, model, optimizer, epoch):

    model.train()
    peer_iter = iter(peer_loader)
    for i, (input, target) in enumerate(train_loader):
        input = torch.autograd.Variable(input.cuda())
        target = target.cuda()
        output = model(input)
        optimizer.zero_grad()
        if epoch >= 10:
            input1 = peer_iter.next()[0]
            input1 = torch.autograd.Variable(input1.cuda())
            output1 = model(input1)
            target2 = torch.randint(0, 14, (target.shape)).cuda()
            loss_reg = activation(-criterion_prob(output, target.long()))
            loss_peer = conjugate(-criterion_prob(output1, target2.long()))
            loss =  loss_reg - loss_peer
        # Cross-Entropy warm up
        else:
            loss = criterion(output, target)
        loss.cuda()
        loss.backward()
        optimizer.step()

# Calculate accuracy
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()
        total += target.size(0)
        output = model(input)
        _, predicted = torch.max(output.detach(), 1)
        correct += predicted.eq(target).cpu().sum().item()
    return 100. * correct / total

# Calculate f-divergence value in the max game
def f_calculate(model, data_loader, peer_loader):
    model.eval()
    f_score = 0
    total_s = 0
    peer_iter = iter(peer_loader)
    for i, (input, target) in enumerate(data_loader):
        input = torch.autograd.Variable(input.cuda())
        target = target.cuda()
        output = model(input)
        input1 = peer_iter.next()[0]
        input1 = torch.autograd.Variable(input1.cuda())
        output1 = model(input1)
        target2 = torch.randint(0, 14, (target.shape)).cuda()
        loss_reg = activation(-criterion_prob(output.detach(), target.long()))
        loss_peer = conjugate(-criterion_prob(output1.detach(), target2.long()))
        score =  loss_peer - loss_reg
        total_s += target.size(0)
        f_score += score * target.size(0)
    return f_score/total_s


def main(writer):
    model_prob = models.resnet50(pretrained=True)
    model_prob.fc = nn.Linear(2048, 14)
    model_prob = model_prob.cuda()
    max_div = -100
    test_acc_result = []
    f_result = []
    for epoch in range(num_epochs):
        print("epoch=", epoch,'r=', args.r)
        learning_rate = 0.002
        if epoch >= 10:
            learning_rate = 5e-5
        if epoch >= 15:
            learning_rate = 1e-5
        if epoch >= 20:
            learning_rate = 5e-6
        if epoch >= 25:
            learning_rate = 1e-6
        if epoch >= 30:
            learning_rate = 5e-7
        if epoch >= 35:
            learning_rate = 1e-7
            
        # We adopted the SGD optimizer
        optimizer_prob = torch.optim.SGD(model_prob.parameters(), momentum=0.9, weight_decay=1e-3, lr=learning_rate)
        train(train_loader=train_loader_noisy, peer_loader = train_peer_loader, model=model_prob, optimizer=optimizer_prob, epoch=epoch)
        print("validating model_prob...")
        
        # Calculate test accuracytest_acc = test(model=model_prob, test_loader=test_loader_)
        test_acc_result.append(test_acc)
        print('test_acc=', test_acc)
        f_div_value = f_calculate(model_prob, valid_loader_noisy, val_peer_loader)
        f_result.append(f_div_value)
        print('f_div_value=', f_div_value)
        
        # Best model is selected by referring to f-div value; the larger, the better!
        if max_div <= f_div_value:
            max_div = f_div_value
            torch.save(model_prob, './trained_models/C1M' + str(args.r) + '_' + str(args.s) + '_' + str(args.divergence) + '_' + str(args.warmup))
            print("saved")
        writer.writerow([epoch,f_div_value, test_acc])
    

def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    # Save statistics
    print("Begin:")
    writer1 = csv.writer(open(f'C1M_{r}_{div}_{args.warmup}.csv','w'))
    writer1.writerow(['Epoch','f_div', 'Test_ACC'])
    main(writer1)
    evaluate('./trained_models/C1M' + str(args.r) + '_' + str(args.s) + '_' + str(args.divergence) + '_' + str(args.warmup))
    print("Traning finished")
