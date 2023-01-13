import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

device = ("cuda" if torch.cuda.is_available() else "cpu")

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

#plt.plot(X_numpy,y_numpy,'bo')
#plt.show()
x = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1) # y(n_samples, ) -> [n_samples,1]

n_samples, n_features = x.shape


def tanh_norm(inputs):
    z = torch.tanh(inputs)
    znorm = z / (torch.sqrt(torch.sum(torch.pow(torch.abs(z), 2), axis=1, keepdims=True)))
    return znorm

InputModule = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
    nn.Tanh(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
    nn.Tanh(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(in_features=120, out_features=84)
).to(device)

OutputModule = nn.Sequential(
    nn.Linear(in_features=84, out_features=10),
).to(device)

def forwardPass_InputModule(x):
    x = InputModule(x)
    return x

def forwardPass_OutputModule(x):
    x = tanh_norm(x)
    x = OutputModule(x)
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

num_classes = 10
n_epochs =  3

# Train Input module
for epoch in range(n_epochs):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        loss_fn = SRS_Loss(forwardPass_InputModule(inputs), targets, num_classes)
        loss_fn.backward()
        optimizer.step()

    if (epoch % 1) == 0:
        print(epoch, loss_fn.item())
        net_repr = forwardPass_InputModule(inputs).detach().cpu()
        net_repr = tanh_norm(net_repr).numpy()

        # fig = plt.figure()
        # plt.title("Input Module Epoch{}".format(epoch))
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(net_repr[:, 0], net_repr[:, 1], net_repr[:, 2], c=targets)
        # plt.show()

"""
"""


def k_mtrx(x0, x1):
    return np.matmul(tanh_norm(x0), tanh_norm(x1).T)

def map_input(x):
    return np.where(
         np.eye(len(x)) == 1,
            np.array(-float("inf"), dtype=np.float32),
            k_mtrx(x, x)
        )


def one_hot_encode(target, n_classes):
    
    if target.ndim > 1:
        target = np.squeeze(target)
    target_onehot = np.zeros((target.shape[0], n_classes))
    # target_onehot[range(target.size(0)), target] = 1
    # target_onehot[range(target.size(0)).long(), target.long()] = 1
    target_onehot[range(target.shape[0]), target.astype(int)] = 1
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

    if np.max(target1) + 1 > n_classes:
        raise ValueError('target1 has at least one invalid entry')
    if np.max(target2) + 1 > n_classes:
        raise ValueError('target2 has at least one invalid entry')

    target1_onehot, target2_onehot = \
        one_hot_encode(target1, n_classes).astype(np.float32), \
        one_hot_encode(target2, n_classes).astype(np.float32)
    ideal = np.matmul(target1_onehot, target2_onehot.T)
    if k_min != 0:
        min_mask = np.full(ideal.shape, k_min)
        ideal = np.where(ideal == 0, min_mask, ideal)

    if k_max != 1:
        max_mask = np.full_like(ideal, k_max)
        ideal = np.where(ideal == 1, max_mask, ideal)
    return ideal
"""