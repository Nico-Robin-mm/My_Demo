# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 05:32:31 2020

@author: 71020
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


np.random.seed(1)

X0 = np.random.normal(2, 1, size=(100, 2)) 
X1 = np.random.normal(-2, 1, size=(100, 2)) 
y0 = np.zeros(X0.shape[0])
y1 = np.ones(X1.shape[0])
X_numpy = np.vstack([X0, X1])
y_numpy = np.hstack([y0, y1])
X = torch.from_numpy(X_numpy).type(torch.FloatTensor)
y = torch.from_numpy(y_numpy).type(torch.LongTensor)

plt.scatter(X0[:, 0], X0[:, 1], color="b")
plt.scatter(X1[:, 0], X1[:, 1], color="r")
plt.show()


class LogisticRegression(nn.Module):
    def __init__(self, d_in, d_hidden_1, d_out):
        super(LogisticRegression, self).__init__()
        self.hidden_1 = nn.Linear(d_in, d_hidden_1)
        self.out = nn.Linear(d_hidden_1, d_out)
        
    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        out = self.out(x)
        return out


LR = 1e-2
EPOCHS = 500    
logistic_regression = LogisticRegression(2, 10, 2)
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.SGD(logistic_regression.parameters(), lr=LR)

for i in range(EPOCHS):
    out = logistic_regression(X)
    loss = loss_func(out, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if i % (EPOCHS // 10) == 0:
        y_hat = torch.max(F.softmax(out, 1), 1)[1].detach().numpy()
        #y_hat = torch.max(out, 1)[1].detach().numpy()
        #y_hat = torch.max(torch.sigmoid(out), 1)[1].detach().numpy()
        accuracy = np.sum(np.array(y_hat==y_numpy, dtype="int")) / len(y_numpy)
        print(i, "loss=%.4f"%loss.item())
        print("accuracy:", accuracy)
        