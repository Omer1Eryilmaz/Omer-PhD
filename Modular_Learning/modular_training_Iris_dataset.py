import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import sklearn
import torch.nn.functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")

data,target_numpy = sklearn.datasets.load_iris(return_X_y=True, as_frame=False)

inputs = torch.from_numpy(data.astype(np.float32))
target = torch.from_numpy(target_numpy.astype(np.float32))

target = target.view(target.shape[0],1) # y(n_samples, ) -> [n_samples,1]

n_samples, n_features = inputs.shape
num_classes = 3

def one_hot_encode(target, n_classes):

        if target.ndim > 1:
            target = torch.squeeze(target)
        target_onehot = torch.zeros((target.shape[0], n_classes))
        # target_onehot[range(target.size(0)), target] = 1
        target_onehot[range(target.size(0)), target.type(torch.long)] = 1
        # target_onehot[range(target.shape[0]), target.type(torch.long)] = 1
        return target_onehot

target_onehot = one_hot_encode(target,num_classes)
print(target_onehot)

def tanh_norm(inputs):
    z = torch.tanh(inputs)
    znorm = z / (torch.sqrt(torch.sum(torch.pow(torch.abs(z), 2), axis=1, keepdims=True)))
    return znorm

InputModule = nn.Sequential(
    nn.Linear(4,32),
    nn.ReLU(),
    nn.Linear(32,2)

).to(device)

OutputModule = nn.Sequential(
    nn.Linear(in_features=2, out_features=3),
).to(device)

def forwardPass_InputModule(x):
    x = InputModule(x)
    return x

def forwardPass_OutputModule(x):
    x = tanh_norm(x)
    x = OutputModule(x)
    #x = nn.Softmax(x)
    
    return x


def SRS_Loss(x, y, num_classes):
    def k_mtrx(x0, x1):
        return torch.matmul(tanh_norm(x0), tanh_norm(x1).T)
    def map_input(x):
        return torch.where(
            torch.eye(len(x)).to(x.device) == 1,
            torch.tensor(-float("inf"), dtype=torch.float32, device=x.device),
            k_mtrx(x, x)
        )
    def one_hot_encode(target, n_classes):

        if target.ndim > 1:
            target = torch.squeeze(target)
        target_onehot = torch.zeros((target.shape[0], n_classes))
        # target_onehot[range(target.size(0)), target] = 1
        target_onehot[range(target.size(0)), target.type(torch.long)] = 1
        # target_onehot[range(target.shape[0]), target.type(torch.long)] = 1
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
        if k_min != 0:  # for tanh activation function
            min_mask = torch.full(ideal.shape, k_min)
            ideal = torch.where(ideal == 0, min_mask, ideal)  # Same class indices are 1 others -1.

        if k_max != 1:  # for ReLU activation function
            max_mask = torch.full_like(ideal, k_max)
            ideal = torch.where(ideal == 1, max_mask, ideal)  # Same class indices are MaxValue others 0.
        return ideal

    k_min = -1.
    xx = map_input(x)
    k_ideal = get_ideal_k_mtrx(y, y, num_classes)  # We are using phi(tanh) in the first module's training.
    return torch.mean(torch.exp(xx[k_ideal == k_min])) # loss is possitive different from the paper.

######################### Train the input module #########################
optimizer = torch.optim.Adam(params=InputModule.parameters(), lr=1e-3)


n_epochs =  600

# Train Input module
for epoch in range(n_epochs):
    inputs = inputs.to(device)
    #target = target.to(device)
    optimizer.zero_grad()
    loss_fn = SRS_Loss(forwardPass_InputModule(inputs),target, num_classes)
    loss_fn.backward()
    optimizer.step()
    if(epoch % 100) == 0:
        print(epoch, loss_fn.item())
        net_repr = forwardPass_InputModule(inputs).detach().cpu()
        net_repr = tanh_norm(net_repr).numpy()
        #plt.scatter(net_repr[:,0], net_repr[:,1],c=target_numpy)
        #plt.title("FirstModuleTraining {}. iteration SRS_Loss: {}".format(iter,loss_fn))
        #plt.show()

# Train Output Module
OutputModule_optimizer = torch.optim.Adam(params=OutputModule.parameters(), lr=1e-3)
OutputModule_loss_fn = torch.nn.CrossEntropyLoss()
# net = nn.Sequential(InputModule, OutputModule)
#stop gradient
for param in InputModule.parameters():
    param.requires_grad = False
n_epochs =  100


for epoch in range(n_epochs):

    InputModule.train()
    OutputModule.train()
    inputs = inputs.to(device)
    target = target.to(device)
    OutputModule_optimizer.zero_grad()
    InputForOutputModule = forwardPass_InputModule(inputs)
    loss_fn = OutputModule_loss_fn(forwardPass_OutputModule(InputForOutputModule), target_onehot)
    loss_fn.backward()
    OutputModule_optimizer.step()
    print("Bu loss{}".format(loss_fn))
 # get final accuracy
    InputModule.eval()
    OutputModule.eval()
    preds = []
    labels = []
    with torch.no_grad():
        
        inputs = inputs.to(device)
        pred = forwardPass_InputModule(inputs)
        pred = forwardPass_OutputModule(pred)
        #pred = F.softmax(pred, dim=1) # Why don't we put this in the OutMod TRAINING?
        _, pred = torch.max(pred, dim=1)
        preds += pred.detach().cpu().numpy().tolist()
        labels += target.detach().cpu().numpy().tolist()
    pred = np.array(preds)# disari al
    target_numpy = np.array(labels)
    print("Epoch {}  acc {:.3f} (%):".format(epoch, np.mean(pred == target_numpy)))
