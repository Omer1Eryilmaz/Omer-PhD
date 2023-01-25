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

class InputModuleClass(nn.Module):
    def __init__(self) -> None:
        super(InputModuleClass,self).__init__()
        self.layer1 = nn.Linear(13,512)
        self.layer2 = nn.Linear(512,2)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
Input_model = InputModuleClass()

class OutputModuleClass(nn.Module):
    def __init__(self) -> None:
        super(OutputModuleClass,self).__init__()
        self.layer3 = nn.Linear(2,512)
        self.layer4 = nn.Linear(512,1)
  
    def forward(self,x):
        x = torch.tanh(x)
        x = (x / (torch.sqrt(torch.sum(torch.pow(torch.abs(x), 2), axis=1, keepdims=True))))
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        return x
  
Output_model = OutputModuleClass()

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

optimizer = torch.optim.SGD(params=Input_model.parameters(), lr=1e-3, momentum=0.9)

print("Model's state dictionary:")
for param_tensor in Input_model.state_dict():
    print(param_tensor, "\t", Input_model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name,"\t", optimizer.state_dict()[var_name])

num_classes = 404 # num of class should be greater equal than sample size 
                 #  to make sure each sample is from a different class.

n_epochs=10000
# Train Input module 
for epoch in range(n_epochs):
    
    X_train  = X_train.to(device)
    y_train = y_train.to(device)
    optimizer.zero_grad()
    loss_fn = SRS_Loss(Input_model(X_train), y_train, num_classes)
    loss_fn.backward()
    optimizer.step()

    
    if (epoch % 1000) == 0:
        print(epoch, loss_fn.item())
        net_repr = Input_model(X_train).detach().cpu()
        net_repr = tanh_norm(net_repr).numpy()
        plt.plot(net_repr[:,0],net_repr[:,1],'bo')
        plt.show()

torch.save(Input_model.state_dict(),os.path.join(PATH,'inputParameters_10000.pt'))

Input_model.load_state_dict(torch.load(os.path.join(PATH,'inputParameters_10000.pt')))
Input_model.eval()

# Train Output Module
OutputModule_optimizer = torch.optim.Adam(params=Output_model.parameters(), lr=1e-3)
OutputModule_loss_fn = torch.nn.MSELoss()


#Do not train input module
for param in Input_model.parameters():
    param.requires_grad = False

n_epochs =  5000

print("*** Test results ***")
plot_loss =[]
plot_pred = []
plot_target =[]

for epoch in range(n_epochs):

    Output_model.train()
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    OutputModule_optimizer.zero_grad()
    InputForOutputModule = Input_model(X_train)
    loss = OutputModule_loss_fn(Output_model(InputForOutputModule), y_train)
    loss.backward()
    OutputModule_optimizer.step()
    plot_loss.append(loss)    
    # get final accuracy
    Input_model.eval()
    Output_model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        X_test = X_test.to(device)
        pred = Input_model(X_test)
        pred = Output_model(pred)   
        #_, pred = torch.max(pred, dim=1) for classification
        preds += pred.detach().cpu().numpy().tolist()
        labels += y_test.detach().cpu().numpy().tolist()
        plot_pred.append(preds)

plot_array = [i.detach().numpy() for i in plot_loss]
plt.plot(range(n_epochs),plot_array )
plt.show()

"""
pred = Input_model(X_test)
pred = Output_model(pred)

y_test = torch.squeeze(y_test)

plt.plot(y_test.detach().numpy()[55:75],'b')
plt.plot(pred.detach().numpy()[55:75],'r')
plt.show()


pred = Input_model(X_test)
pred = Output_model(pred)

y_test = torch.squeeze(y_test)

plt.plot(y_test.detach().numpy()[0:20],'-b',label="target")
plt.plot(pred.detach().numpy()[0:20],'-r',label="prediction")
plt.title("Boston housing test set modular training")
plt.xlabel("house samples")
plt.ylabel("house price 10000$")
plt.legend(loc="upper right")
plt.show()

plot_array = [i.detach().numpy() for i in plot_loss]
plt.plot(plot_array,'-k',label="loss")
plt.title("Boston housing test set modular training")
plt.xlabel("number of epochs")
plt.legend(loc="upper right")
print("the last epoch loss:{:.2f}".format(plot_array[-1]))
plt.show()

#torch.save(Output_model.state_dict(),os.path.join(PATH,'outputParameters_10000.pt'))

"""