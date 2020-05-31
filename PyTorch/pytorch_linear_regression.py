# -*- coding: utf-8 -*-
"""
Created on Sun May 31 22:10:44 2020

@author: 71020
"""


import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


data = torch.linspace(-1, 1, 100)
X = data.view(-1, 1)
y = X.pow(2) + 0.2*torch.rand(size=X.size())

plt.scatter(X, y)
plt.show()

class Net(nn.Module):
    
    def __init__(self, d_in, d_hid_1, d_hid_2, d_out):
        super(Net, self).__init__()
        self.hidden_1 = nn.Linear(d_in, d_hid_1)
        self.hidden_2 = nn.Linear(d_hid_1, d_hid_2)
        self.predict = nn.Linear(d_hid_2, d_out)
        
    def forward(self, x):
        x1 = F.relu(self.hidden_1(x))
        x2 = F.relu(self.hidden_2(x1))
        y = self.predict(x2)
        return y
    
    
net = Net(1, 10, 10, 1)
loss_func = nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr=0.2)

plt.ion()

for i in range(400):
    y_hat = net(X)
    loss = loss_func(y_hat, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i % 20 == 0:
        plt.cla()
        plt.scatter(X, y)
        plt.plot(X, y_hat.detach().numpy(), "r-")
        plt.text(0.75, 0, "Loss=%.4f"%loss.item())
        plt.pause(0.1)
plt.ioff()
plt.show()
    
    
