import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import sklearn
import pandas as pd
data = load_iris()
data.target[[110, 65, 50]] # 0-50 arasi 0, 51-100 1, 101-149 2 bunlar indis
list(data.target_names)

data,target = sklearn.datasets.load_iris(return_X_y=True, as_frame=False)
from sklearn.preprocessing import StandardScaler
x = data[0:50,(1,2)]  #works that way ?
y = data[0:50,(0,1)]

x=x.reshape(-1, 1) 
print(x.shape)
y=y.reshape(-1, 1) 
print(y.shape)

#plt.plot(x,y,'ro')
#plt.show()

x = torch.tensor(x, requires_grad=True,dtype=torch.float)
y = torch.tensor(y, requires_grad=True,dtype=torch.float)


#2 Fully connected network   -------------------
# loss function MSE (y-activation)^2
# activation function ReLU
# Self> model1 = Neural_Net(13,1)
print("data size {}".format(x.shape))

class Neural_Net(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super(Neural_Net,self).__init__()
        self.Linear1 = nn.Linear(input_size,hidden_size1)
        self.Linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.Linear3 = nn.Linear(hidden_size2,output_size)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        output = self.Linear1(x)
        output = self.activation(output)
        output = self.Linear2(output)
        output = self.activation(output)
        output = self.Linear3(output)

        # Consider adding activation function
        return output

model = Neural_Net(1,20,20,1)
# Training
criterion = nn.L1Loss()
epochs = 1000
optimizer = torch.optim.SGD(model.parameters(),lr=0.003,momentum=0.9)
plot_loss=[]

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
    plot_loss.append(loss)

plot_array = [i.detach().numpy() for i in plot_loss]
plt.plot(range(epochs),plot_array )
plt.show()

print(f'model first 5 output: {output[5:10]} ')
print(f'label: {y[5:10]}')
print(f'loss: {loss}')