import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

InputModule = nn.Sequential(
    nn.Linear(13,512),
    nn.ReLU(),
    nn.Linear(512,2)).to(device)

OutputModule = nn.Sequential(
    nn.Linear(2, 512),
    nn.ReLU(),
    nn.Linear(512,1),
).to(device)

def forwardPass_InputModule(x):
    x = InputModule(x)
    return x

def forwardPass_OutputModule(x):
    x = tanh_norm(x)
    x = OutputModule(x)
    return x

def SRS_Loss(x,y, num_classes):
    def k_mtrx(x0,x1):
        return torch.matmul(tanh_norm(x0),tanh_norm(x1).T)
    def map_input(x):
        return torch.where(
            torch.eye(len(x)) == 1,
                torch.tensor(-float("inf"), dtype=torch.float32),
                k_mtrx(x,x)
        )
    def one_hot_encode(target, n_classes):
        
        if target.ndim > 1:
            target = torch.squeeze(target)
        target_onehot = torch.zeros((target.shape[0], n_classes))
        # target_onehot[range(target.size(0)), target] = 1
        target_onehot[range(target.size(0)), target.type(torch.long)] = 1
        #target_onehot[range(target.shape[0]), target.type(torch.long)] = 1
        return target_onehot

    def get_ideal_k_mtrx(target1, target2, n_classes):
        
        k_min = -1.
        k_max = 1.
        if n_classes < 2:
            raise ValueError('You need at least 2 classes')

        if len(target1.shape) > 2:
            raise ValueError('target1 has too many dimensions')
        if len(target2.shape) > 2:
            raise ValueError('target2 has too many dimensions')

        if torch.max(target1) + 1 > n_classes:
            raise ValueError('target1 has at least one invalid entry')
        if torch.max(target2) + 1 > n_classes:
            raise ValueError('target2 has at least one invalid entry')

        target1_onehot, target2_onehot = \
            one_hot_encode(target1, n_classes).type(torch.float32), \
            one_hot_encode(target2, n_classes).type(torch.float32)
        ideal = torch.matmul(target1_onehot, target2_onehot.T)
        if k_min != 0: # for tanh activation function
            min_mask = torch.full(ideal.shape, k_min)
            ideal = torch.where(ideal == 0, min_mask, ideal) #Same class indices are 1 others -1.

        if k_max != 1: # for ReLU activation function
            max_mask = torch.full_like(ideal, k_max)
            ideal = torch.where(ideal == 1, max_mask, ideal) # Same class indices are MaxValue others 0.
        return ideal

######################### Train the input module #########################

    k_min = -1.
    x = tanh_norm(x) # *** important change ***
    xx = map_input(x)
    k_ideal = get_ideal_k_mtrx(y, y, num_classes) #We are using phi(tanh) in the first module's training.
    return torch.mean(torch.exp(xx[k_ideal == k_min]))

optimizer = torch.optim.Adam(params=InputModule.parameters(), lr=1e-3)


num_classes=500 # num of class should be greater equal than sample size 
                 #  to make sure each sample is from a different class.
n_epochs=50

# Train Input module 
for epoch in range(n_epochs):
    
    X_train  = X_train.to(device)
    y_train = y_train.to(device)
    optimizer.zero_grad()
    loss_fn = SRS_Loss(forwardPass_InputModule(X_train), y_train, num_classes)
    loss_fn.backward()
    optimizer.step()

    
    if (epoch % 50) == 0:
        print(epoch, loss_fn.item())
        net_repr = forwardPass_InputModule(X_train).detach().cpu()
        net_repr = tanh_norm(net_repr).numpy()
        plt.plot(net_repr[:,0],net_repr[:,1],'bo')
        plt.show()
"""
Alternatively you can save trained input module and then load its parameters and just train the output module.
https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
 

# Train Output Module
OutputModule_optimizer = torch.optim.Adam(params=OutputModule.parameters(), lr=1e-3)
OutputModule_loss_fn = torch.nn.MSELoss()


#Do not train input module
for param in InputModule.parameters():
    param.requires_grad = False

n_epochs =  5000

print("*** Test results ***")
for epoch in range(n_epochs):

    InputModule.train()
    OutputModule.train()
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    OutputModule_optimizer.zero_grad()
    InputForOutputModule = forwardPass_InputModule(X_train)
    loss_fn = OutputModule_loss_fn(forwardPass_OutputModule(InputForOutputModule), y_train)
    loss_fn.backward()
    OutputModule_optimizer.step()
        
    # get final accuracy
    InputModule.eval()
    OutputModule.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        X_test = X_test.to(device)
        pred = forwardPass_InputModule(X_test)
        pred = forwardPass_OutputModule(pred)   
        _, pred = torch.max(pred, dim=1)
        preds += pred.detach().cpu().numpy().tolist()
        labels += y_test.detach().cpu().numpy().tolist()
preds = np.array(preds)
labels = np.array(labels)
"""