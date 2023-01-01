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
print("data size {}".format(X.shape))

class Neural_Net(nn.Module):
    def __init__(self,input_size,output_size):
        super(Neural_Net,self).__init__()
        self.Linear1 = nn.Linear(input_size,output_size)
        #self.Linear2 = nn.Linear(hidden_size,hidden_size)
        #self.Linear3 = nn.Linear(hidden_size,output_size)
        self.activation = nn.ReLU()

    def forward(self,x):
        output = self.Linear1(x)
        output = self.activation(output)
        #output = self.Linear2(output)
        #output = self.activation(output)
        #output = self.Linear3(output)
        # Consider adding activation function
        return output

model = Neural_Net(13,1)
output = model(X)

# Training
criterion = nn.MSELoss()

epochs = 1000

optimizer = torch.optim.SGD(model.parameters(),lr=0.003,momentum=0.9)
plot_loss=[]

'''
for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
    plot_loss.append(loss)

plt.plot(range(60,epochs),plot_loss[60:epochs])
plt.show()
print(plot_loss[999])

'''


