import torch
import torch.nn as nn
from sklearn.datasets import load_boston
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

# 1- Boston House dataset download
#       Optional PCA 
#2- Fully connected network
#3- Train
#4- Print loss

#1- Boston House dataset download
batch_size=50
X, y = load_boston(return_X_y=True)
X = X[:,[2,5]]
X = preprocessing.normalize(X)
train_loader = DataLoader(dataset=X, batch_size=batch_size, shuffle=True)

X= next(iter(train_loader))
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

model = Neural_Net(2,512,1)


# Training
criterion = nn.L1Loss()
epochs = 200
optimizer = torch.optim.SGD(model.parameters(),lr=0.03,momentum=0.8)
plot_loss=[]

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(y,output)
    loss.backward()
    optimizer.step()
    plot_loss.append(loss)


plot_array = [i.detach().numpy() for i in plot_loss]
plt.plot(range(epochs),plot_array )
plt.show()

#mean_error=np.mean(np.abs(output.detach().numpy()-y.detach().numpy()))
#np.mean(y)-output
#y = [i.detach().numpy() for i in y]