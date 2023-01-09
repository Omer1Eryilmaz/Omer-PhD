import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data,target = load_iris(return_X_y=True, as_frame=False)
print(type(data))
num_classes = 3


def one_hot_encode(target, n_classes):
    
    if target.ndim > 1:
        target = np.squeeze(target)
    target_onehot = np.zeros((target.shape[0], n_classes))
    target_onehot[range(target.shape[0]), target.astype(int)] = 1
    return target_onehot

target = one_hot_encode(target,num_classes)

data = torch.tensor(data,requires_grad=True,dtype=torch.float) # 150,4
target = torch.tensor(target,requires_grad=True,dtype=torch.float) 

class MyModule(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh() 
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.outputs = []

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            
            if layer != self.layers[-1]:
                x = self.activation(x)
            else:
                x = self.tanh(x)
            self.outputs.append(x)

        return x / (torch.sqrt(torch.sum(torch.pow(torch.abs(x), 2), axis=1, keepdims=True)))

model = MyModule([4, 512, 512, 3])# 1,16- 16-32 , 32- 2 
output = model(data)


kernel = torch.matmul(output,output.T)
ideal = torch.matmul(target,target.T) 

kernel = torch.tensor(kernel,requires_grad=True,dtype=torch.float)
ideal = torch.tensor(ideal,requires_grad=True,dtype=torch.float)

criterion = nn.CrossEntropyLoss()
"""
def srs_loss(kernel,ideal):
    output = torch.mean(torch.exp(kernel[ideal!=1]))
    return torch.tensor(output,requires_grad=True)
"""

optimizer = torch.optim.Adam(params = model.parameters(), lr=1e-3)

epochs = 100
plot_loss = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(kernel,ideal)
    loss.backward()
    optimizer.step()

    plot_loss.append(loss)

plot_array = [i.detach().numpy() for i in plot_loss]
plt.plot(range(epochs),plot_array )
plt.show()

print(f'model first 5 output: {output[60:65]} ')
print(f'label: {target[60:65]}')
print(f'loss: {loss}')

