# This is a test model for "class_type_modular_regression.py"

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os


PATH = os.getcwd()

device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training
# Load dataset
bc = datasets.load_boston()
X,y = bc.data, bc.target


n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1234)

#Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#Convert to Torch tensor 
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# model 
def tanh_norm(inputs):
    z = torch.tanh(inputs)
    znorm = z / (torch.sqrt(torch.sum(torch.pow(torch.abs(z), 2), axis=1, keepdims=True)))
    return znorm

class modelClass(nn.Module):
    def __init__(self) -> None:
        super(modelClass,self).__init__()
        self.layer1 = nn.Linear(13,512)
        self.layer2 = nn.Linear(512,1)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
model = modelClass()

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

n_epochs = 5000
train_loss =[]
test_loss =[]
for epoch in range(n_epochs):

    model.train()
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    train_loss.append(loss)    
    # get final accuracy
    model.eval()
    preds = []
    
    with torch.no_grad():
        X_test = X_test.to(device)
        pred = model(X_test).numpy()   
        target = y_test.detach().numpy()
        m = len(target)
        loss = (1/(m)) * np.sum((pred - target) ** 2)
        test_loss.append(loss)

plot_train_loss = [i.detach().numpy() for i in train_loss]
plot_test_loss = [i for i in test_loss]
print("Epoch 5000 training loss {} test loss {}".format(train_loss[-1],test_loss[-1]))
plt.plot(range(n_epochs),plot_train_loss,'-b',label="train loss")
plt.plot(range(n_epochs),plot_test_loss,'-r',label = "test loss")
plt.title("Two layer model standard training")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.show()
