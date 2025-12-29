import pandas as pd 
import numpy as np
from sklearn import model_selection
import torch
from collections import Counter

from utils import rlr_ann_validate

import os
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore", "Initializing zero-element tensors is a no-op")

from dtuimldmtools import train_neural_net
from dtuimldmtools.statistics.statistics import correlated_ttest

#Load clean dataset from data folder
path = os.path.abspath(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
path = os.path.join(path, "data", "clean_dataset.csv")
#path = "~/Desktop/Master/MachineLearning/02450_Machine_Learning_and_Data_Mining/project/project_2/data/clean_dataset.csv"
#data = pd.read_csv(path)

data = pd.read_csv(path)

#Extract features
#X = np.array(data.values[:,1:5],  dtype=float)  
X = np.log10(np.array(data.values[:,1:3],  dtype=float))
#Rings
#y = np.array(data.values[:,-2],  dtype=float).squeeze() # --> number of rings (31 classes)
y = np.log10(np.array(data.values[:,4:5],  dtype=float).squeeze())

#Standardization
N = len(y)
X = (X - np.ones((N, 1)) * X.mean(axis=0)) / X.std(axis=0)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
#attributeNames = list(data.columns[1:5])
attributeNames = list(data.columns[1:3])
attributeNames =  ["Offset"] + attributeNames
M = len(attributeNames)

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
#lambdas = np.power(10.0, range(-5, 9))
small_range = np.linspace(0.6, 1, 5, endpoint=True)  # Adjust the number of points if needed

# Generate values from 10 upwards
large_range = np.array([(i+1)*10 for i in range(10)])

# Concatenate the arrays
lambdas = np.concatenate((small_range, large_range))

hs = list(range(1,15))

# Initialize variables
opt_and_err = {"Outer fold" : [], # --> dictionary to create the final table (akin to table 1)
               "λ*" : [],
               "Eλ" : [],
               "h*": [],
               "Eh": [],
               "baseline": []}

Error_train = np.empty((K)) # --> unregularized linear regression
Error_test = np.empty((K))

Error_train_nofeatures = np.empty((K)) # --> baseline model
Error_test_nofeatures = np.empty((K))

Error_train_lambda = np.empty(K) # --> regularized linear regression
Error_test_lambda = np.empty(K)

Error_train_h = np.empty(K) # --> neural network prediction 
Error_test_h = np.empty(K)

mu = np.empty((K, M - 1)) # --> the means of the train set for each outer fold
sigma = np.empty((K, M - 1)) # --> the std ...

w_rlr = np.empty((M, K)) # --> the regularized weigths of the regressions
w_noreg = np.empty((M, K))

# define loss function of the neural network
loss_fn = torch.nn.MSELoss() 

# where to contain the difference in perfomance (error rlr - error NN)
r_rlr_vs_ANN = []
r_rlr_vs_base = []
r_ANN_vs_base = []

k = 0
for train_index, test_index in CV.split(X, y):
    print(f"\nEntering OUTER FOLD: {k}")
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # go through the inner folds
    (
        opt_val_err_lambda, # --> lowest error of a linear regression
        opt_lambda, # --> lambda associated with the lowest test error
        train_err_vs_lambda, # --> record of the train error of all the rlr 
        test_err_vs_lambda, # --> record of the test error of all the rlr 

        opt_val_err_h, # same as for lambda but for the NN with different number of hidden units
        opt_h,
        train_err_vs_h,
        test_err_vs_h,
    ) = rlr_ann_validate(X_train, y_train, lambdas, hs)
    

    # store mean and std of this specific train set
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    #Standardization on this partition
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]
    
    # Estimate weights for the optimal value of lambda, on entire training set
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the best performin lambda, on entire training set
    w_rlr[:, k] = np.linalg.solve(XtX + opt_lambda, Xty).squeeze()
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
                
    # Compute mean squared error with regularization
    Error_train_lambda[k] = (np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0])
    Error_test_lambda[k] = (np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0])

    # Compute mean squared error without regularization
    Error_train[k] = (np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0])
    Error_test[k] = (np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0])

    # Compute mean squared error without features (baseline)
    Error_train_nofeatures[k] = (np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0])
    Error_test_nofeatures[k] = (np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0])
    
    # cast dataset to Tensor to allow interaction with the NN
    X_train = torch.Tensor(X_train)
    y_train = torch.unsqueeze(torch.Tensor(y_train), 1)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    # Define the model with the optimal number of hidden units found in the inner fold
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, opt_h),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(opt_h, 1),  # n_hidden_units to 1 output neuron
    )

    # Train the model 
    with redirect_stdout(open(os.devnull, 'w')):
                net, Error_train_h[k], learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train,
                y=y_train,
                n_replicates=1,
                max_iter=1000
            )
                
    # Compute MSE of the NN prediction
    se = (y_test.float() - torch.squeeze(net(X_test).float())) ** 2  # squared error
    Error_test_h[k] = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean

    # store each of the variables for the final column in its place
    opt_and_err["Outer fold"].append(k + 1)
    opt_and_err["Eλ"].append(round(Error_test_lambda[k], 5))
    opt_and_err["λ*"].append(opt_lambda)
    opt_and_err["Eh"].append(round(Error_test_h[k], 5))
    opt_and_err["h*"].append(opt_h)
    opt_and_err["baseline"].append(round(Error_test_nofeatures[k].item(), 5))

    r_rlr_vs_ANN.append(abs(Error_test_lambda[k] - Error_test_h[k]))
    r_rlr_vs_base.append(abs(Error_test_lambda[k] - Error_test_nofeatures[k]))
    r_ANN_vs_base.append(abs(Error_test_h[k] - Error_test_nofeatures[k]))

    k += 1

mean = lambda x: sum(x)/len(x)

# find optimal values of lambda and h...
opt_lambda = mean(opt_and_err["λ*"])
opt_h = mean(opt_and_err["h*"])

# ...and estimate their error
Error_train_nofeatures_mean = np.mean(Error_train_nofeatures, axis=0)
Error_test_nofeatures_mean = np.mean(Error_test_nofeatures, axis=0)

# get the mean error of the unregularized linear regression
Error_train_mean = np.mean(Error_train, axis=0)
Error_test_mean = np.mean(Error_test, axis=0)

# build summary df (Table 1) 
df_summary = pd.DataFrame(opt_and_err)
df_summary.to_csv('./data/df_summary.csv', index=False)

# calculate correlated t-test to see if one model performs significantly better than the other
alpha = 0.05
rho = 1/K
p_rlr_vs_ANN, CI_rlr_vs_ANN = correlated_ttest(r_rlr_vs_ANN, rho, alpha=alpha)
p_rlr_vs_base, CI_rlr_vs_base = correlated_ttest(r_rlr_vs_base, rho, alpha=alpha)
p_ANN_vs_base, CI_ANN_vs_base = correlated_ttest(r_ANN_vs_base, rho, alpha=alpha)

# save ttest results
with open('ttest_results.txt', 'w') as file:
    file.write(f"alpha: {alpha}\n")
    file.write(f"rho: {rho}\n")
    file.write(f"r rlr vs ANN: {r_rlr_vs_ANN}\n")
    file.write(f"p-value rlr vs ANN: {p_rlr_vs_ANN}\n")
    file.write(f"Confidence Interval rlr vs ANN: {CI_rlr_vs_ANN}\n\n")
    
    file.write(f"r rlr vs base: {r_rlr_vs_base}\n")
    file.write(f"p-value rlr vs base: {p_rlr_vs_base}\n")
    file.write(f"Confidence Interval rlr vs base: {CI_rlr_vs_base}\n\n")

    file.write(f"r ANN vs base: {r_ANN_vs_base}\n")
    file.write(f"p-value ANN vs base: {p_ANN_vs_base}\n")
    file.write(f"Confidence Interval ANN vs base: {CI_ANN_vs_base}\n\n")