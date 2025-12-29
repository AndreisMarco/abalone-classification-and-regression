import numpy as np
from sklearn import model_selection
import torch
from tqdm import tqdm

import os
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore", "Initializing zero-element tensors is a no-op")

from dtuimldmtools import train_neural_net

def rlr_ann_validate(X, y, lambdas, hs, cvf=10):
    """Validate regularized linear regression model using 'cvf'-fold cross validation.
    Find the optimal lambda (minimizing validation error) from 'lambdas' list.
    The loss function computed as mean squared error on validation set (MSE).
    Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
    average weight values for all lambdas, MSE train&validation errors for all lambdas.
    The cross validation splits are standardized based on the mean and standard
    deviation of the training set when estimating the regularization strength.

    Parameters:
    X       training data set
    y       vector of values
    lambdas vector of lambda values to be validated
    cvf     number of crossvalidation folds

    Returns:
    opt_val_err         validation error for optimum lambda
    opt_lambda          value of optimal lambda
    mean_w_vs_lambda    weights as function of lambda (matrix)
    train_err_vs_lambda train error as function of lambda (vector)
    test_err_vs_lambda  test error as function of lambda (vector)
    """
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M, cvf, len(lambdas)))

    train_error_lambda = np.empty((cvf, len(lambdas)))
    test_error_lambda = np.empty((cvf, len(lambdas)))
    train_error_h = np.empty((cvf, len(hs)))
    test_error_h = np.empty((cvf, len(hs)))

    loss_fn = torch.nn.MSELoss()
    
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X, y):
        print(f"Entering INNER FOLD: {f}")
        # splits for regression
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # splits for ANN
        X_train_h = torch.Tensor(X_train)
        y_train_h = torch.unsqueeze(torch.Tensor(y_train), 1)
        X_test_h = torch.Tensor(X_test)
        y_test_h = torch.Tensor(y_test)

        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in tqdm(range(0, len(lambdas)), desc="TESTING LAMBDAS"):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error_lambda[f, l] = np.power(y_train - X_train @ w[:, f, l].T, 2).mean(axis=0            )
            test_error_lambda[f, l] = np.power(y_test - X_test @ w[:, f, l].T, 2).mean(axis=0)

        for h in tqdm(range(0, len(hs)), desc="TESTING HS"):
            # Define the model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(h, 1),  # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
            )

            with redirect_stdout(open(os.devnull, 'w')):
                net, train_error_h[f, h], learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train_h,
                y=y_train_h,
                n_replicates=1,
                max_iter=10000
            )

            # Determine errors and errors
            se = (y_test_h.float() - torch.squeeze(net(X_test_h).float())) ** 2  # squared error
            test_error_h[f, h] = (sum(se).type(torch.float) / len(y_test_h)).data.numpy()  # mean

        f = f + 1

    opt_val_err_lambda = np.min(np.mean(test_error_lambda, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error_lambda, axis=0))]
    train_err_vs_lambda = np.mean(train_error_lambda, axis=0)
    test_err_vs_lambda = np.mean(test_error_lambda, axis=0)

    opt_val_err_h = np.min(np.mean(test_error_h, axis=0))
    opt_h = hs[np.argmin(np.mean(test_error_h, axis=0))]
    train_err_vs_h = np.mean(train_error_h, axis=0)
    test_err_vs_h = np.mean(test_error_h, axis=0)

    return (
        opt_val_err_lambda,
        opt_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
        opt_val_err_h,
        opt_h,
        train_err_vs_h,
        test_err_vs_h,
    )
