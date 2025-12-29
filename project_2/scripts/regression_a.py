import pandas as pd 
import numpy as np
from sklearn import model_selection
import os

from dtuimldmtools import rlr_validate

from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    plot,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
    xticks,
    yticks,
    tight_layout
)

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
X = np.log10(np.array(data.values[:,1:4],  dtype=float))
#Rings
#y = np.array(data.values[:,-2],  dtype=float).squeeze() # --> number of rings (31 classes)
y = np.log10(np.array(data.values[:,4:5],  dtype=float).squeeze())

#Standardization
N = len(y)
X = (X - np.ones((N, 1)) * X.mean(axis=0)) / X.std(axis=0)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
#attributeNames = list(data.columns[1:5])
attributeNames = list(data.columns[1:4])
attributeNames =  ["Offset"] + attributeNames
M = len(attributeNames)

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
#lambdas = np.power(10.0, range(-5, 9))
small_range = np.linspace(0, 1, 10, endpoint=True)  # Adjust the number of points if needed

# Generate values from 10 upwards
large_range = np.array([(i+1)*1 for i in range(100)])

# Concatenate the arrays
lambdas = np.concatenate((small_range, large_range))

Error_train = np.empty((K)) # --> unregularized linear regression
Error_test = np.empty((K))

Error_train_nofeatures = np.empty((K)) # --> baseline model
Error_test_nofeatures = np.empty((K))

Error_train_lambda = np.empty((K,len(lambdas))) # --> regularized linear regression
Error_test_lambda = np.empty((K,len(lambdas)))

mu = np.empty((K, M - 1)) # --> the means of the train set for each outer fold
sigma = np.empty((K, M - 1)) # --> the std ...

w_rlr = np.empty((M, K)) # --> the regularized weigths of the regressions models
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV.split(X, y):
    #print(f"\nEntering OUTER FOLD: {k}")
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # go through the inner folds
    (
        opt_val_err_lambda, # --> lowest error of a linear regression
        opt_lambda, # --> lambda associated with the lowest test error
        mean_w_vs_lambda,
        train_err_vs_lambda, # --> record of the train error of all the rlr 
        test_err_vs_lambda, # --> record of the test error of all the rlr 
    ) = rlr_validate(X_train, y_train, lambdas)

    # store the error vs lambda
    Error_train_lambda[k] = train_err_vs_lambda
    Error_test_lambda[k] = test_err_vs_lambda

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = (
        np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    )
    Error_test_nofeatures[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # store mean and std of this specific train set
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    #Standardization on this partition
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]
    
    # Estimate weights for the optimal value of lambda, on entire training set
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the best performing lambda, on entire training set
    w_rlr[:, k] = np.linalg.solve(XtX + opt_lambda, Xty).squeeze()
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
                
    # Compute mean squared error with regularization
    Error_train_lambda[k] = (np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0])
    Error_test_lambda[k] = (np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0])

    # Compute mean squared error without regularization
    Error_train[k] = (np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0])
    Error_test[k] = (np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0])

    # Display the results for the last cross-validation fold
    if k == K - 1:
        lamdas_error_plot = figure(figsize=(12, 8))
        #title(r"Optimal lambda: $10^{%d}$" % np.log10(opt_lambda), fontsize=14)
        title(F"Optimal lambda: {opt_lambda}", fontsize=16)
        loglog(
        #plot(
            lambdas, train_err_vs_lambda.T, "b.-",
            lambdas, test_err_vs_lambda.T, "r.-",
            # lambdas, np.ones(len(lambdas)) * Error_train[k], ".--",
            # lambdas, np.ones(len(lambdas)) * Error_test[k], ".--",
            # lambdas, np.ones(len(lambdas)) * Error_train_nofeatures[k], ".:",
            # lambdas, np.ones(len(lambdas)) * Error_test_nofeatures[k], ".:",
        )
        xlabel("Regularization factor", fontsize=16)
        ylabel("Squared error (crossvalidation)", fontsize=16)
        legend(["Train error", "Val error", "unreg train error", "unreg val error", "baseline train error", "baseline test error"], fontsize=14)
        xticks(fontsize=14)
        yticks(fontsize=14)
        tight_layout()
        grid()
        lamdas_error_plot.savefig("./plots/error_vs_lamdas_smaller_range.pdf", format="pdf")

        rlr_prediction_plot = figure(figsize=(12, 8))
        plot(y_test, X_test @ w_rlr[:, k], ".")
        xlabel("Whole weight (true)", fontsize=16)
        ylabel("Whole weight (estimated)", fontsize=16)

        max_range = max(abs(y_test).max(), abs((X_test @ w_rlr[:, k]).max()))
        plot([0, max_range], [0, max_range], "r--", label="x = y-")
        grid()
        #rlr_prediction_plot.savefig("./plots/rlr_prediction.pdf", format="pdf")

        weights_plot = figure(figsize=(12, 8))
        semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")
        xlabel("Regularization factor", fontsize=16)
        ylabel("Mean Coefficient Values", fontsize=16)
        tight_layout()
        grid()
        #weights_plot.savefig("./plots/weights_plot.pdf", format="pdf")

    k += 1

mean = lambda x: sum(x)/len(x)

# ...and estimate their error
Error_train_nofeatures_mean = np.mean(Error_train_nofeatures, axis=0)
Error_test_nofeatures_mean = np.mean(Error_test_nofeatures, axis=0)

# get the mean error of the unregularized linear regression
Error_train_mean = np.mean(Error_train, axis=0)
Error_test_mean = np.mean(Error_test, axis=0)


