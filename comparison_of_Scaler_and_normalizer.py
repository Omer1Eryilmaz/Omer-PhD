import sklearn.preprocessing as preprocessing 

from sklearn.preprocessing import StandardScaler
# Normalizer and normalize do exactly the same operations but work with diff. syntex
# Normalization is the process of scalling individual samples to have unit norm.
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize

import numpy as np
# Normalization is the process of scalling individual samples to have unit norm.
# Standardization converts data into in a shape with zero mean and unit variance.

# Normalize the given data X
X = [[ 1., -1.,  2.],
    [ 2.,  0.,  0.],  
    [ 0.,  1., -1.]]
X = np.array(X)
print('X given data : \n {}'.format(X))
# Use of normalize
X_normalized = normalize(X, norm ='l2')
print('X_normalized using \'normalize\' : \n {}'.format(X_normalized))

# Use of Normalizer
normalizer = Normalizer().fit(X)
X_normalized_by_Normalizer = normalizer.transform(X)
print('X_normalized_by_Normalizer using \'Normalizer\' : \n {}'.format(X_normalized_by_Normalizer))

# Note that each sample features are normalized to have unit norm 
# so features having large scales may dominate the result.

# Scale the data
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
print("X_scaled :\n {}".format(X_scaled))
print("X mean :\n {} X scale:\n {}".format(scaler.mean_,scaler.scale_))

# use this scale to scale a mock test data
print(" *** If that makes sense ??? ***")
test_data = [[3., -4., 1.],
            [5., 0., 0.],
            [0., -1., 6.]]
test_data = np.array(test_data)
test_data_Scaled = scaler.transform(test_data)
print("test_data_Scaled :\n {}".format(test_data_Scaled))

# Create a scale for the test data

test_scaler = preprocessing.StandardScaler().fit(test_data)
test_Scaled_by_test_scaler = test_scaler.transform(test_data)
print("test_Scaled_by_test_scaler :\n {}".format(test_Scaled_by_test_scaler))
"""
# Scaling features to a range
min_max_scaler = preprocessing.MinMaxScaler()
X_min_max = min_max_scaler.fit_transform(X)
print('restricted scaling [0 1] of X  :\n {}'.format(X_min_max))
"""

