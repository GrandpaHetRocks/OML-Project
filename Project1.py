# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:38:14 2021

@author: Ayush
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from Dataset import load_dataset, getImage
import random
import math
import matplotlib.pyplot as plt

class Arguments():
    def __init__(self):
        self.images = 10000
        self.clients = 10
        self.rounds = 1
        self.epochs = 200
        self.local_batches = 64
        self.lr = 0.007
        self.torch_seed = 0 #same weights and parameters whenever the program is run
        self.log_interval = 64
        self.iid = 'iid'
        self.split_size = int(self.images / self.clients)
        self.samples = self.split_size / self.images 
        self.use_cuda = False
        self.save_model = True
 
args = Arguments()

use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

global_train, global_test, train_group, test_group = load_dataset(args.clients, args.iid)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
global_test_dataset = datasets.MNIST('./', train=False, download=True, transform=transform)
global_test_loader = DataLoader(global_test_dataset, batch_size=args.local_batches, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.fc1 = nn.Linear(4*4*10, 10) #noniid:50 iid:5
        self.fc2 = nn.Linear(10, 10) #50

    def forward(self, x):
        #x=self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
client={}
    
torch.manual_seed(args.torch_seed)
client['model'] = Net().to(device)
client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr)

clients=[client]
ac=[]
for inx, client in enumerate(clients):  #return actual image set for each client
    trainset_ind_list = list(train_group[inx]) 
    client['trainset'] = getImage(global_train, trainset_ind_list, args.local_batches)
    client['testset'] = getImage(global_test, list(test_group[inx]), args.local_batches)
    client['samples'] = len(trainset_ind_list) / args.images #useful while taking weighted average
    
for round in range(1,args.rounds+1):
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(client['trainset']): 
            data,target=data,target
            # data = data.send(client['hook'])
            # target = target.send(client['hook'])
            
            #train model on client
            data, target = data.to(device), target.to(device) #send data to cpu/gpu (data is stored locally)
            output = client['model'](data)
            loss = F.nll_loss(output, target)
            loss.backward()
            client['optim'].step()
            
            if batch_idx % args.log_interval == 0:
                # loss = loss.get() 
                print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    "client1",
                    epoch, batch_idx * args.local_batches, len(client['trainset']) * args.local_batches, 
                    100. * batch_idx / len(client['trainset']), loss))
    client['model'].eval()    #no need to train the model while testing
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in global_test_loader:
            if(use_cuda):
                data,target=data.cuda(),target.cuda()
                #model.cuda()
            else:
                data, target = data.to(device), target.to(device)
            output = client['model'](data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(global_test_loader.dataset)

    print('\nTest set: Average loss for {} model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        'client1', test_loss, correct, len(global_test_loader.dataset),
        100. * correct / len(global_test_loader.dataset)))
    ac.append(100. * correct / len(global_test_loader.dataset))

print(ac)