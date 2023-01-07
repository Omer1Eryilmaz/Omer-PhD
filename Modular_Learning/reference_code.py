import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data,target = load_iris(return_X_y=True, as_frame=False)
print(type(data))


data = torch.tensor(data,requires_grad=True,dtype=torch.float) # 150,4
target = torch.tensor(target,requires_grad=True,dtype=torch.float) 


class MyModule(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.outputs = []

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            
            if layer != self.layers[-1]:
                x = self.activation(x)
            self.outputs.append(x)

        return x

model = MyModule([4, 16, 32, 3])# 1,16- 16-32 , 32- 2 

output = model(data)
print("output:{}".format(output.shape))

