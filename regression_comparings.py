from sklearn.datasets import load_boston
# **** Need to scale the data **** Means are different.
from sklearn.preprocessing import StandardScaler
# z = (x - u)/s, x=data sample, u= mean, s= number of data sample.
X,y = load_boston(return_X_y=True)
#print(X[1,:])
scaler = StandardScaler()
scaler.fit(X) #data
X_scaled = scaler.transform(X)
#print(X_scaled[1,:])