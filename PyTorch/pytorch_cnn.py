# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 01:24:13 2020

@author: 71020
"""


import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(1)

# Hyper parameters
EPOCHS = 2 
BATCH_SIZE = 50
LR = 1e-2
DOWNLOAD_MNIST = False # dataset need download or not?

# MNIST
training_data = torchvision.datasets.MNIST(
    root="./mnist/", # location
    train=True,
    transform=torchvision.transforms.ToTensor(),
    # trans PIL.Image or numpy.ndarray to torch.FloatTensor(C*H*W), and normalize
    download = DOWNLOAD_MNIST
    )

# plot one example
plt.imshow(training_data.data[0].numpy(), cmap="gray")
plt.title("%i"%training_data.targets[0])
plt.show()

# data loader
train_loader = data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up training
test_data = torchvision.datasets.MNIST(root="./mnist/", train=False)
test_X = test_data.data.view(-1, 1, 28, 28).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.targets[:2000]

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Sequential( # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1, # input height
                out_channels=16, # n_filters
                kernel_size=5, # filter size
                stride=1, # filter step
                padding=2 # (5-1)/2
                ), # output shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape (16, 14, 14)
            )
        self.conv2 = nn.Sequential( # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2), # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2) # output shape (32, 7, 7)
            )
        self.hidden = nn.Linear(32 * 7 * 7, 100) # fully connected layer
        self.out = nn.Linear(100, 10)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x2.view(x2.size(0), -1)
        x4 = F.relu(self.hidden(x3))
        output = self.out(x4)
        return output


cnn = Cnn()
print(cnn)        

optim = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if step % 50 == 0:
                test_output = cnn(test_X)
                pred_y = torch.max(F.softmax(test_output), 1)[1].detach().numpy()
                accuracy = float((pred_y == test_y.numpy()).astype(int).sum()) / float(test_y.size(0))
                print("Epoch:", epoch, "| train loss: %.4f"%loss.item(), "| test accuracy: %.2f"%accuracy)
    test_output2 = cnn(test_X[:10])
    pred_y2 = torch.max(F.softmax(test_output), 1)[1].detach().numpy()[:10]
    print(pred_y2, "predict numbers")
    print(test_y[:10].numpy(), "real numbers")
                


































