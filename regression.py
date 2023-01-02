import torch
import torch.nn as nn
from sklearn.datasets import load_boston
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

# 1- Boston House dataset download
#       Optional PCA 
#2- Fully connected network
#3- Train
#4- Print loss

#1- Boston House dataset download
X, y = load_boston(return_X_y=True)
X = preprocessing.normalize(X)
X = torch.tensor(X, requires_grad=True,dtype=torch.float)
y = torch.tensor(y, requires_grad=True,dtype=torch.float)


#2- Fully connected network
# 13 feature > 512> 512 > 1
# loss function MSE (y-activation)^2
# activation function ReLU
# Self> model1 = Neural_Net(13,1)
print("data size {}".format(X.shape))

class Neural_Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Neural_Net,self).__init__()
        self.Linear1 = nn.Linear(input_size,hidden_size)
        self.Linear2 = nn.Linear(hidden_size,hidden_size)
        self.Linear3 = nn.Linear(hidden_size,hidden_size)
        self.Linear4 = nn.Linear(hidden_size,output_size)
        self.activation = nn.ReLU()

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

model = Neural_Net(13,32,1)


# Training
criterion = nn.MSELoss()

epochs = 100

optimizer = torch.optim.SGD(model.parameters(),lr=0.03,momentum=0.9)
plot_loss=[]

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
    plot_loss.append(loss)
    

plt.plot(range(epochs),plot_loss)
plt.show()

"""
# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=Tr
"""


