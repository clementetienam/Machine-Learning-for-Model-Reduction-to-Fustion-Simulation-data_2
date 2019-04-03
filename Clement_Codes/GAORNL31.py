# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:20:11 2019

@author: mjkiqce3
"""
import numpy as np
print('cluster with X and y')
X = open("inpiecewise.out") #533051 by 28
X = np.fromiter(X,float)
X = np.reshape(X,(10000,2), 'F') 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
(scaler.fit(X))
X=(scaler.transform(X))
y = open("outpiecewise.out") #533051 by 28
y = np.fromiter(y,float)
y = np.reshape(y,(10000,1), 'F')  

ydami=y


ytest=y
outputtest=ytest

numrowstest=len(outputtest)

inputtrainclass=X
#
outputtrainclass=y
matrix=np.concatenate((X,y), axis=1)
#

#
xtest=X
inputtest=xtest

domain= [(-1,1),(-1,1)]
n_bins=(50,50)
input_values=X
target_values=y

from sklearn.model_selection import train_test_split
(input_train,input_test,target_train,target_test)=train_test_split(input_values,target_values,test_size=0.20,random_state=42)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, n_bins_xy, domain_xy):
        super(Net, self).__init__()
        self.coeffs=nn.Parameter(torch.zeros(n_bins_xy[0],2))
        self.domain_xy=np.array(domain_xy)
        self.n_bins_xy=np.array(n_bins_xy)
        self.bin_width_xy=(self.domain_xy[:,1]-self.domain_xy[:,0])/self.n_bins_xy
    def single_coord_func(self,input,coord):
        indices = np.floor((input - self.domain_xy[coord,0]) / self.bin_width_xy[coord])
        indices = np.clip(indices, 0, self.n_bins_xy[coord] - 1)
        return torch.cat([ self.coeffs[indices[row], coord] for row in range(len(input)) ])
    def forward(self,input):
        batch_size=input.size(0)
        indices=np.zeros((batch_size,2))
        return self.single_coord_func(input[:,0], 0) + self.single_coord_func(input[:,1], 1)


model= Net(n_bins_xy =n_bins,domain_xy=domain)
lossFunc=nn.MSELoss()
minibatch_size=32
num_epochs=30
optimizer=optim.Adam(model.parameters(),lr=0.001)
train_losses = []; test_losses = []
test_target_var=Variable(torch.from_numpy(np.stack(target_test)))
print('starting training')
for epoch in range(num_epochs):
    all_indices=np.arange(len(target_train))
    np.random.shuffle(all_indices)
    model.train()
    train_loss=0
    train_steps=0
    for start in range(0,len(target_train),minibatch_size):
        stop=min(start+minibatch_size,len(target_train))
        indices=all_indices[start:stop]
        optimizer.zero_grad()
        this_inputs=input_train[indices]
        input_var=Variable(torch.from_numpy(this_inputs))
        output = model.forward(input_var)
        this_target=Variable(torch.from_numpy(target_train[indices]))
        loss=lossFunc.forward(output,this_target)
        train_loss +=loss.data(0)
        loss.backward()
        optimizer.step()
        train_steps +=1
    train_loss/=train_steps
    train_losses.append(train_loss)
    model.eval()
    output=model.forward(Variable(torch.from_numpy(input_test)))
    test_loss=lossFunc.forward(output,test_target_var).data(0)
    test_losses.append(test_loss)
    print("epoch %d:train loss: %.4f test loss: %.4f"%(epoch +1,train_loss,test_loss))
    
    
    
    
    
    
    
    
