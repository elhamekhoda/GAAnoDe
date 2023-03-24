# this script read is data files and returns normalized versions of them

import os
import sys
from sklearn.preprocessing import StandardScaler
from scipy.special import logit, expit
import numpy as np

#this fucntion is used to load the data from arguments
# def get_outer_data(args):
#     outerdata_test = np.load(sys.argv[1])
#     outerdata_train = np.load(sys.argv[2])

#     nFeat = 6
#     outerdata_train = outerdata_train[outerdata_train[:,nFeat+1]==0]
#     outerdata_test = outerdata_test[outerdata_test[:,nFeat+1]==0]

#     data_train = outerdata_train[:,1:nFeat+1]
#     data_test = outerdata_test[:,1:nFeat+1]
#     data = np.concatenate((data_train, data_test), axis=0)

#     cond_data_train = outerdata_train[:,0]
#     cond_data_test = outerdata_test[:,0]
#     cond_data = np.concatenate((cond_data_train, cond_data_test), axis=0)

#     return data, cond_data

def minmax_norm_data(six_feature_data):
    data = six_feature_data 
    delta_shift = 1.0e-3
    nFeat = 6
    x_max = np.empty(nFeat)
    x_min = np.empty(nFeat)
    for i in range(0,data.shape[1]):
        x_max[i] = np.max(data[:,i])
        x_min[i] = np.min(data[:,i])
        if np.abs(x_max[i]) > 0: 
            data[:,i] = ((data[:,i] - x_min[i]) + delta_shift) / ((x_max[i]- x_min[i]) + 2*delta_shift)
        else:
            pass

    return data, x_max, x_min

# this function is used to do standard normalization on the data
def standard_norm_data(six_feature_data):
    data = six_feature_data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler

# min max normalization on conditional data
def minmax_norm_cond_data(cond_data):
    delta_shift = 1.0e-3
    cond_max = np.max(cond_data)
    cond_min = np.min(cond_data)
    if np.abs(cond_max) > 0: 
        cond_data = ((cond_data - cond_min) + delta_shift) / ((cond_max-cond_min) + 2*delta_shift)
    else:
        pass
    return cond_data

# this function is used to do standard normalization on the conditional data
def standard_norm_cond_data(cond_data):
    scaler = StandardScaler()
    cond_data = np.reshape(cond_data, [-1, 1])
    cond_data = scaler.fit_transform(cond_data)
    return cond_data, scaler

# this function is used to do logit normalization on the data. for both data and conditional data.
def logit_norm(data):
    data = logit(data)
    return data

# this function is used to do expit normalization on the data. for both data and conditional data
def expit_norm(data):
    data = expit(data)
    return data

# this fucntion does reverse min max normalization on the data
def rev_minmax_data(new_data, old_data_min, old_data_max):
    delta_shift = 1.0e-3
    data = new_data
    for i in range(0,data.shape[1]):
        data[:,i] = (data[:,i] * ((old_data_max[i]-old_data_min[i]) + 2*delta_shift)) - delta_shift + old_data_min[i]
    return data

#this function does reverse standard normalization on the data
def rev_standard_data(new_data, scaler):
    data = scaler.inverse_transform(new_data)
    return data



