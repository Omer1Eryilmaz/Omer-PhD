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

x = data[0:50,1]
y = data[0:50,0]
print(x.shape)
print(y.shape)

plt.plot(x,y,'ro')
plt.show()

x = torch.tensor(x, requires_grad=True,dtype=torch.float)
x = x.view(50,1)

y = torch.tensor(y, requires_grad=True,dtype=torch.float)

y = y.view(50,1)
# simdi bunda ayni seyleri deneriz

class Neural_Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Neural_Net,self).__init__()
        self.Linear1 = nn.Linear(input_size,hidden_size)
        self.Linear2 = nn.Linear(hidden_size,hidden_size)
        self.Linear3 = nn.Linear(hidden_size,hidden_size)
        self.Linear4 = nn.Linear(hidden_size,output_size)
        self.activation = nn.Sigmoid()

    def forward(self,x):
        output = self.Linear1(x)
        output = self.activation(output)
        output = self.Linear2(output)
        output = self.activation(output)
        output = self.Linear3(output)
        output = self.activation(output)
        output = self.Linear4(output)
        # Consider adding activation function
        return output

model = Neural_Net(1,512,1) # belki giris 150 ??


# Training
criterion = nn.MSELoss()

epochs = 100

optimizer = torch.optim.SGD(model.parameters(),lr=0.3,momentum=0.9)
plot_loss=[]
all_outputs = []
for epoch in range(epochs):

    optimizer.zero_grad()
    output = model(x) # yeni data kucuk x
    loss = criterion(y,output)
    loss.backward()
    optimizer.step()
    plot_loss.append(loss)
    all_outputs.append(output)

plt.plot(range(epochs),plot_loss)
#plt.plot(range(epochs),output)
plt.show()



