# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:56:37 2020

@author: 71020
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(446)
np.random.seed(446)

# =============================================================================
# d_in = 3
# d_out = 4
# linear_module = nn.Linear(d_in, d_out)
# 
# example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
# transformed = linear_module(example_tensor)
# print("example_tensor", example_tensor.shape)
# print("transformed", transformed.shape)
# print("w:", linear_module.weight)
# print("b", linear_module.bias)
# 
# activation_fn = nn.ReLU()
# example_tensor = torch.tensor([-1.0, 1.0, 0.0])
# activated = activation_fn(example_tensor)
# print(example_tensor)
# print(activated)
# =============================================================================
# =============================================================================
# d_in = 3
# d_hidden = 4
# d_out = 1
# model = torch.nn.Sequential(
#     nn.Linear(d_in, d_hidden),
#     nn.Tanh(),
#     nn.Linear(d_hidden, d_out),
#     nn.Sigmoid()
#     )
# example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
# transformed = model(example_tensor)
# print("transformed:",transformed.shape)
# 
# params = model.parameters()
# for param in params:
#     print(param)
# 
# mse_loss_fn = nn.MSELoss()
# input_2 = torch.tensor([[0., 0, 0]])
# target = torch.tensor([[1., 0, -1]])
# loss = mse_loss_fn(input_2, target)
# print(loss)
# =============================================================================

# =============================================================================
# model = nn.Linear(1, 1)
# X_simple = torch.tensor([[1.]])
# y_simple = torch.tensor([[2.]])
# 
# optim = torch.optim.SGD(model.parameters(), lr=1e-2)
# mse_loss_fn = nn.MSELoss()
# 
# y_hat = model(X_simple)
# print(model.weight)
# loss = mse_loss_fn(y_hat, y_simple)
# optim.zero_grad()
# loss.backward()
# optim.step()
# print("model params after:", model.weight)
# =============================================================================

# =============================================================================
# d = 2
# n = 50
# X = torch.randn(n, d)
# true_w = torch.tensor([[-1.0], [2.0]])
# y = X @ true_w + torch.randn(n, 1) * 0.1
# 
# lr = 0.1
# 
# linear_module = nn.Linear(d, 1, bias=False)
# 
# loss_func = nn.MSELoss()
# 
# optim = torch.optim.SGD(linear_module.parameters(), lr = lr)
# print("iter,\tloss,\tw")
# 
# for i in range(20):
#     y_hat = linear_module(X)
#     loss = loss_func(y_hat, y)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     print("\n{},\t{:.2f},\t{}".format(i, loss.item(), linear_module.weight.view(2).detach().numpy()))
#     print("true w\t\t", true_w.view(2).numpy())
#     print("estimated w\t", linear_module.weight.view(2).detach().numpy())
# =============================================================================

# =============================================================================
# d = 2
# n = 50
# X = torch.randn(n, d)
# true_w = torch.tensor([[-1.0], [2.0]])
# y = X @ true_w + torch.randn(n, 1) * 0.1
# lr = 0.01
# linear_module = nn.Linear(d, 1)
# loss_func = nn.MSELoss()
# optim = torch.optim.SGD(linear_module.parameters(), lr=lr)
# for i in range(200):
#     rand_index = np.random.choice(n)
#     x = X[rand_index]
#     y_hat = linear_module(x)
#     loss = loss_func(y_hat, y[rand_index])
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
# print(true_w)
# print(linear_module.weight)
# =============================================================================

d = 1
n = 200
X = torch.rand(n, d)
y = 4*torch.sin(np.pi * X)*torch.cos(6*np.pi*X**2)

lr = 0.05
epochs = 6000
hidden_1 = 32
hidden_2 = 32
d_out = 1
neural_network = nn.Sequential(
    nn.Linear(d, hidden_1),
    nn.Tanh(),
    nn.Linear(hidden_1, hidden_2),
    nn.Tanh(),
    nn.Linear(hidden_2, d_out)
    )
loss_func = nn.MSELoss()
optim = torch.optim.SGD(neural_network.parameters(), lr=lr)
for i in range(epochs):
    y_hat = neural_network(X)
    loss = loss_func(y_hat, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if i % (epochs // 10) == 0:
        print(i, "loss", loss.item())

momentum = 0.9

neural_network_2 = nn.Sequential(
    nn.Linear(d, hidden_1),
    nn.Tanh(),
    nn.Linear(hidden_1, hidden_2),
    nn.Tanh(),
    nn.Linear(hidden_2, d_out)
    )
optim_2 = torch.optim.SGD(neural_network_2.parameters(), lr=lr, momentum=momentum)
for i in range(epochs):
    y_hat2 = neural_network_2(X)
    loss_2 = loss_func(y_hat2, y)
    optim_2.zero_grad()
    loss_2.backward()
    optim_2.step()
    
    if i % (epochs // 10) == 0:
        print(i, "loss_2", loss_2.item())


















