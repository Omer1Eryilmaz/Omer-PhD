import torch
import torch.nn as nn
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt

# 1- Boston House dataset download
#       Optional PCA 
#2- Fully connected network
#3- Train
#4- Print loss

#1- Boston House dataset download
X, y = load_boston(return_X_y=True)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
plt.plot(X[:,1],y,'bo')
plt.show()

#2- Fully connected network
# 13 feature > 512> 512 > 1
# loss function MSE (y-activation)^2
# activation function ReLU
# Self> model1 = Neural_Net(13,1)

class Neural_Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Neural_Net,self).__init__()
        self.Linear1 = nn.Linear(input_size,hidden_size)
        self.Linear2 = nn.Linear(hidden_size,hidden_size)
        self.Linear3 = nn.Linear(hidden_size,output_size)
        self.activation = nn.ReLU()

    def forward(self,x):
        output = self.Linear1(x)
        output = self.activation(output)
        output = self.Linear2(output)
        output = self.activation(output)
        output = self.Linear3(output)
        # Consider adding activation function
        return output

model = Neural_Net(13,512,1)
output = model(X)
print(output.shape)

anda
anda
anda
a