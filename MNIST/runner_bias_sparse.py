# Import libraries
import numpy as np
import os
import csv
import math
import torch.nn.parallel
import pickle
import argparse
from tqdm import tqdm
from noise_data_mnist import *
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
num_classes = 10
num_epochs = 150
CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

CE = nn.CrossEntropyLoss().cuda()

opt = parser.parse_args()

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

elif div == 'Reverse-KL':
    def activation(x): return -torch.mean(-torch.exp(x))
    
    def conjugate(x): return -torch.mean(-1. - x)  # remove log

elif div == 'Jeffrey':
    def activation(x): return -torch.mean(x)
    
    def conjugate(x): return -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)

elif div == 'Squared-Hellinger':
    def activation(x): return -torch.mean(1. - torch.exp(x))
    
    def conjugate(x): return -torch.mean((1. - torch.exp(x)) / (torch.exp(x)))

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
        
criterion_prob = ProbLossStable()
criterion_prob.cuda()

# Training
def train(train_loader, peer_loader, model, optimizer, epoch):

    model.train()
    peer_iter = iter(peer_loader)
    for i, (idx, input, target) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            continue
        warmup_epoch = args.warmup
        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())
        output = model(input)
        optimizer.zero_grad()
        # After warm-up epochs, switch to optimizing f-divergence functions
        if epoch >= warmup_epoch:
        
            # Estimate E_Z [g(Z)] where Z follows the joint distribution of h, noisy Y;
            # g is the activation function
            prob_reg = -criterion_prob(output, target)
            loss_regular = activation(prob_reg)
            
            # Bias correction of sparse noise case for the activation term
            for k in range(5):
                target_tmp1 = target * 0. + 2 * k
                target_tmp2 = target * 0. + 2 * k + 1
                loss_regular -= transition_matrix[2 * k][2 * k + 1] * activation(-criterion_prob(output, target_tmp2.long()))
                loss_regular -= transition_matrix[2 * k + 1][2 * k] * activation(-criterion_prob(output, target_tmp1.long()))
            
            #Estimate E_Z [f^*(g(Z))] where Z follows the product of marginal distributions of h, noisy Y;
            # f^*(g) is the conjugate function;
            input1 = peer_iter.next()[1]
            input1 = torch.autograd.Variable(input1.cuda())
            output1 = model(input1)
            target2 = torch.randint(0, 10, (target.shape)).cuda()
            prob_peer = -criterion_prob(output1, target2)
            loss_peer = conjugate(prob_peer)
            
            # Bias correction of sparse noise case for the conjugate term
            for k in range(5):
                target_tmp1 = target2 * 0. + 2 * k
                target_tmp2 = target2 * 0. + 2 * k + 1
                loss_peer -= transition_matrix[2 * k][2 * k + 1] * conjugate(-criterion_prob(output1, target_tmp2.long()))
                loss_peer -= transition_matrix[2 * k + 1][2 * k] * conjugate(-criterion_prob(output1, target_tmp1.long()))
            loss = loss_regular - loss_peer
        # Use CE loss for the warm-up.
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

    for i, (idx, input, target) in enumerate(test_loader):
        input = torch.Tensor(input).cuda()
        target = torch.autograd.Variable(target).cuda()

        total += target.size(0)
        output = model(input)
        _, predicted = torch.max(output.detach(), 1)
        correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / total

    return accuracy
    

# Calculate f-divergence value in the max game
def f_calculate(model, data_loader, peer_loader):
    model.eval()
    f_score = 0
    peer_iter = iter(peer_loader)
    for i, (idx, input, target) in enumerate(data_loader):
        if idx.size(0) != batch_size:
            continue
        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())
        output = model(input)
        prob_reg = -criterion_prob(output.detach(), target)
        loss_regular = activation(prob_reg)
        for k in range(5):
            target_tmp1 = target * 0. + 2 * k
            target_tmp2 = target * 0. + 2 * k + 1
            loss_regular -= transition_matrix[2 * k][2 * k + 1] * activation(-criterion_prob(output.detach(), target_tmp2.long()))
            loss_regular -= transition_matrix[2 * k + 1][2 * k] * activation(-criterion_prob(output.detach(), target_tmp1.long()))
        input1 = peer_iter.next()[1]
        input1 = torch.autograd.Variable(input1.cuda())
        output1 = model(input1)
        target2 = torch.randint(0, 10, (target.shape)).cuda()
        prob_peer = -criterion_prob(output1.detach(), target2)
        loss_peer = conjugate(prob_peer)
        for k in range(5):
            target_tmp1 = target2 * 0. + 2 * k
            target_tmp2 = target2 * 0. + 2 * k + 1
            loss_peer -= transition_matrix[2 * k][2 * k + 1] * conjugate(-criterion_prob(output1.detach(), target_tmp2.long()))
            loss_peer -= transition_matrix[2 * k + 1][2 * k] * conjugate(-criterion_prob(output1.detach(), target_tmp1.long()))
        score = loss_peer - loss_regular
        f_score += score * target.size(0)
    return f_score/10000


def main(writer):
    model_prob = CNNModel().cuda()
    best_prob_acc = 0
    max_f = -100
    val_acc_noisy_result = []
    train_acc_result = []
    test_acc_result = []
    f_result = []
    f_test_result = []
    # Dataloader for peer samples, which is used for the estimation of the marginal distribution
    peer_train = peer_data_train(batch_size=args.batchsize, img_size=(32, 32))
    peer_val = peer_data_val(batch_size=args.batchsize, img_size=(32, 32))
    peer_test = peer_data_test(batch_size=args.batchsize, img_size=(32, 32))
    
    for epoch in range(num_epochs):
        print("epoch=", epoch,'r=', args.r)
        learning_rate = 1e-3
        if epoch > 20:
            learning_rate = 5e-4
        elif epoch > 40:
            learning_rate = 1e-4
        elif epoch > 60:
            learning_rate = 5e-5
        elif epoch > 80:
            learning_rate = 1e-5
        elif epoch > 100:
            learning_rate = 5e-6
        elif epoch > 120:
            learning_rate = 1e-6
        elif epoch > 140:
            learning_rate = 5e-7
        
        optimizer_prob = torch.optim.Adam(model_prob.parameters(), lr=learning_rate)
        train(train_loader=train_loader_noisy, peer_loader = peer_train, model=model_prob, optimizer=optimizer_prob, epoch=epoch)
        print("validating model_prob...")
        
        # Training acc is calculated via noisy training data
        train_acc = test(model=model_prob, test_loader=train_loader_noisy)
        train_acc_result.append(train_acc)
        print('train_acc=', train_acc)
        
        # Validation acc is calculated via noisy validation data
        valid_acc = test(model=model_prob, test_loader=valid_loader_noisy)
        val_acc_noisy_result.append(valid_acc)
        print('valid_acc_noise=', valid_acc)
        
        # Calculate test accuracy
        test_acc = test(model=model_prob, test_loader=test_loader_)
        test_acc_result.append(test_acc)
        print('test_acc=', test_acc)
        f_div_value = f_calculate(model_prob, valid_loader_noisy, peer_val)
        f_result.append(f_div_value)
        print('f_div_value=', f_div_value)
        f_test = f_calculate(model_prob, test_loader_, peer_test)
        f_test_result.append(f_test)
        print('f_test_value=', f_test)
       
        # Best model is selected by referring to f-div value; the larger, the better!
        if f_div_value >= max_f:
            max_f = f_div_value
            torch.save(model_prob, './trained_models/BIAS_SPARSE' + str(args.r) + '_' + str(args.s) + '_' + str(args.divergence) + '_' + str(args.warmup))
            print("saved, f-div value increases.")
                
        writer.writerow([epoch, train_acc, valid_acc, test_acc, f_div_value, f_test])



def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_)
    print('test_acc=', test_acc)



if __name__ == '__main__':
    # Save statistics
    print("Begin:")
    writer1 = csv.writer(open(f'BIAS_SPARSE_result_{r}_{div}_{args.warmup}.csv','w'))
    writer1.writerow(['Epoch', 'Training Acc', f'Val_Noisy_Acc', 'Test_ACC', 'f_div', 'f_test'])
    main(writer1)
    evaluate('./trained_models/BIAS_SPARSE' + str(args.r) + '_' + str(args.s) + '_' + str(args.divergence) + '_' + str(args.warmup))
    print("Traning finished")
