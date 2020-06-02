# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 05:32:31 2020

@author: 71020
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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


"""
* dataset (Dataset): 加载数据的数据集
* batch_size (int, optional): 每批加载多少个样本
* shuffle (bool, optional): 设置为“真”时,在每个epoch对数据打乱.（默认：False）
* sampler (Sampler, optional): 定义从数据集中提取样本的策略,返回一个样本
* batch_sampler (Sampler, optional): like sampler, but returns a batch of indices at a time 返回一批样本. 与atch_size, shuffle, sampler和 drop_last互斥.
* num_workers (int, optional): 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
* collate_fn (callable, optional): 合并样本列表以形成一个 mini-batch.  #　callable可调用对象
* pin_memory (bool, optional): 如果为 True, 数据加载器会将张量复制到 CUDA 固定内存中,然后再返回它们.
* drop_last (bool, optional): 设定为 True 如果数据集大小不能被批量大小整除的时候, 将丢掉最后一个不完整的batch,(默认：False).
* timeout (numeric, optional): 如果为正值，则为从工作人员收集批次的超时值。应始终是非负的。（默认：0）
* worker_init_fn (callable, optional): If not None, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: None)．

"""

class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0), \
            "data_tensor and target_tensor must be equal in size"
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.data_tensor.size(0)


BATCH_SIZE = 20
DATASET = TensorDataset(X, y)
dataloader = DataLoader(dataset=DATASET, batch_size=BATCH_SIZE, shuffle=True)


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
EPOCHS = 51    
logistic_regression = LogisticRegression(2, 10, 2)
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.SGD(logistic_regression.parameters(), lr=LR)

for i in range(EPOCHS):
    for step, (b_X, b_y) in enumerate(dataloader): 
        out = logistic_regression(X)
        loss = loss_func(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    
        if i % 10 ==0 and step % 9 == 0:
            # i:0-50, step:0-9  （len(X)=200， batch_size=20， 200/20=10， 即0-9）           
            y_hat = torch.max(F.softmax(out, 1), 1)[1].detach().numpy()
            #y_hat = torch.max(out, 1)[1].detach().numpy()
            #y_hat = torch.max(torch.sigmoid(out), 1)[1].detach().numpy()
            accuracy = np.sum(np.array(y_hat==y_numpy, dtype="int")) / len(y_numpy)
            print("Epoch:%d"%i, "step:%i"%step, "loss=%.4f"%loss.item())
            print("accuracy:", accuracy)
            
