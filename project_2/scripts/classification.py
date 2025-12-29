from dtuimldmtools.statistics.statistics import correlated_ttest
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode, ttest_rel, t
from tqdm import tqdm

FIGSIZE = (16, 9)
COLOR_BASELINE  = "skyblue"
COLOR_KNN       = "lightgreen"
COLOR_LOGREG    = "lightcoral"
FONT_SIZE       = 16
TITLE_FONT_SIZE = 20
SYMBOL_SIZE     = 150

# The content and implementations in this file are based of and inspired by examples from the 02450toolbox and the course book:
# - exercise 6.3.2 (Cross-validation)
# - exercise 7_1_1 (KNN)
# - exercise 7_3_1 (SETUP II)
# - exercise 9.2.2 (Logistic Regression)
# - Algorithm 6 (Two-level cross-validation) <-- from the course book

### UTILS ###
def get_project_path():
    # Get the path of the project
    path = os.path.join(
        os.path.dirname(                    # One directory lower
            os.path.dirname(                # One directory lower
                os.path.abspath(__file__)   # Path of this file
            )
        )
    )
    return path

def create_path_if_not_exists(folder_name):
    # Create a folder if it does not exist
    path = os.path.join(
        get_project_path(), 
        folder_name
    )

    if not os.path.exists(path):
        os.makedirs(path)

    return path    

### DATA ###
def get_data_from_folder(folder_name, file_name):
    # Load clean dataset from data folder
    path = os.path.join(
        get_project_path(), 
        folder_name, 
        file_name
    )

    # Read data from csv file
    data = pd.read_csv(path)

    return data   

def get_features(data):
    # Extract the features from data set
    X_df = data.drop(["Sex", "Rings", "Number_of_rings"], axis=1)
    X = np.array(X_df,  dtype=float)

    header = X_df.columns.tolist()

    # Get groupings of the entries based on no. of rings
    labels = ["Young (0-8)", "Mature (9-10)", "Old (11-29)"]
    conditions = [
        data["Rings"] <= 8,
        (data["Rings"] >= 9) & (data["Rings"] <= 10),
        data["Rings"] >= 11
    ]
    label = np.select(conditions, labels)
    classdict = dict(zip(labels, [0,1,2]))
    y = np.asarray([classdict[value] for value in label])
    #y = np.array(data.values[:,-1])

    # Standardization of features
    N, M = X_df.shape
    X = (X - np.ones((N, 1)) * X.mean(axis=0)) / X.std(axis=0)

    return X, X_df, y, N, M, header, labels

def save_df_to_csv(df, folder_name, file_name):
    # Save the DataFrame to a csv file
    df.to_csv(
        os.path.join(create_path_if_not_exists(folder_name), file_name),
        header = True, 
        index = False
    )

def save_df_to_latex(df, folder_name, file_name):
    # Save the DataFrame to a latex file
    df.to_latex(
        bold_rows= True,
        header= True,
        buf = os.path.join(create_path_if_not_exists(folder_name), file_name),
        index = False
    )

def save_weights_to_file(w_class_0, w_class_1, w_class_2, K1, header, labels):
    # Save the weights to a csv file and a latex file    
    data = []
    for i in range(0, len(header)):
        data.append([
            np.round(w_class_0 / K1, 4)[i], 
            np.round(w_class_1 / K1, 4)[i], 
            np.round(w_class_2 / K1, 4)[i]
        ])       

    # Create DataFrame from the list of rows
    df = pd.DataFrame(data).T
    df.columns = header

    # Add the class labels as the first column
    df.insert(0, "Class (rings)", [labels[0], labels[1], labels[2]])

    # Save the DataFrame to a csv file
    save_df_to_csv(df, "results", "weights.csv")
    save_df_to_latex(df, "results", "weights.tex")      

### STATISTICS ###
# Setup II - Correlated t-test for cross-validation
def coorelated_t_test_cross_validation(gen_err_1, gen_err_2, alpha = 0.05):
    # Use library function to calculate the t-test 
    t_score, p_value = ttest_rel(gen_err_1, gen_err_2, alternative = "two-sided")
    
    # Calculate the confidence interval manually
    z_arr = gen_err_1 - gen_err_2
    n = len(z_arr)
    z_tilde = np.mean(z_arr) 
    sigma_tilde = np.sqrt(sum((z_arr - z_tilde) * (z_arr - z_tilde) / (n*(n-1))))
    CI_l, CI_u = t.interval(1 - alpha, n-1, loc = z_tilde, scale = sigma_tilde)

    return round(t_score, 4), p_value, round(CI_l, 4), round(CI_u, 4)

### PLOTS ###
def create_knn_plot(L, error_knn_inner_2):
    fig, ax = plt.subplots(figsize = FIGSIZE)
    
    # X and Y values
    x = np.arange(1, L+1, 1)
    y = error_knn_inner_2.mean(0) * 100
    
    # Plot a black line connecting the points
    ax.plot(x, y, color = "black", linewidth = 2)
    
    # Scatter plot with colored points
    scatter = ax.scatter(x, y, color = COLOR_KNN, edgecolor = "black", s = SYMBOL_SIZE, zorder = 3)
    
    # Add labels
    plt.xlabel("No. of neighbors", fontsize = TITLE_FONT_SIZE)
    plt.ylabel("Classification error rate (%)", fontsize = TITLE_FONT_SIZE)
    
    # Set tick font size
    ax.tick_params(axis = "both", labelsize = FONT_SIZE)

    return fig

def create_logistic_regression_plot(lambdas, error_lr_inner_2):
    fig = plt.figure(figsize = FIGSIZE)

    plt.semilogx(
        lambdas, error_lr_inner_2.mean(0) * 100, "-o", 
        color = "black",
        markerfacecolor = COLOR_LOGREG, 
        markeredgecolor = "black",
        markersize = SYMBOL_SIZE / 10
    )
    plt.xlabel("Regularization strength, $\log_{10}(\lambda)$", fontsize = TITLE_FONT_SIZE)
    plt.ylabel("Classification error rate (%)", fontsize = TITLE_FONT_SIZE)
    
    # Set tick font size
    plt.tick_params(axis = "both", labelsize = FONT_SIZE)

    return fig

def create_boxplot_classification(error_base, error_knn_outer_1, error_lr_outer_1):
    fig, ax = plt.subplots(figsize = FIGSIZE)
    
    # Prepare data for boxplots
    boxes = [error_base, error_knn_outer_1, error_lr_outer_1]
    boxes_df = pd.DataFrame(boxes).T
    labels = ["Baseline", "K-Nearest Neighbor", "Logistic Regression"]
    
    # Create boxplot
    boxplot = ax.boxplot(boxes_df, patch_artist = True)
    
    # Define colors for each box
    colors = [COLOR_BASELINE, COLOR_KNN, COLOR_LOGREG]
    
    # Apply colors to each box in the boxplot
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)

    # Set each median marker to black
    for median in boxplot["medians"]:
        median.set_color("black")

    # Set labels and title
    ax.set_ylabel("Generalization Error", fontsize = TITLE_FONT_SIZE)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize = TITLE_FONT_SIZE)
    
    # Set tick font size
    ax.tick_params(axis = "y", labelsize = FONT_SIZE)
    
    return fig

def save_plot_to_folder(folder_name, file_name, dpi = 300):

    # Define the path to save the plot
    path = os.path.join(
        create_path_if_not_exists(folder_name), 
        file_name
    )

    # Save plot to file
    plt.savefig(path, dpi = dpi, bbox_inches="tight")

if __name__ == "__main__":
    
    # Load the data
    data = get_data_from_folder("data", "clean_dataset.csv")

    # Extract the features from data set
    X, X_df, y, N, M, header, labels = get_features(data)

    # Set parameters for cross validation
    # Maximum number of neighbors
    L = 40

    # Outer folds
    K1 = 10

    # Inner folds
    K2 = 10

    # Values of lambda
    lambdas = np.logspace(-3, 2, 10)
    
    # Set up Two-level cross validation - use random state to ensure reproducibility of results
    CV_outer_1 = model_selection.KFold(n_splits = K1, shuffle = True, random_state = 123)
    CV_inner_2 = model_selection.KFold(n_splits = K2, shuffle = True, random_state = 123)

    # Initialize error arrays for baseline
    error_base = np.zeros((K2))

    # Initialize error arrays for logistic regression
    error_lr_outer_1 = np.zeros((K1))
    error_lr_inner_2 =  np.zeros((K2, len(lambdas)))
    error_lr_min = np.zeros(K2)
    opt_lambda = np.zeros(K2)

    # Initialize weights for logistic regression
    w_class_0 = np.zeros(M)
    w_class_1 = np.zeros(M)
    w_class_2 = np.zeros(M)

    # Initialize error arrays for KNN
    error_knn_outer_1 = np.zeros((K1))
    error_knn_inner_2 = np.zeros((K2, L))

    # Initialize arrays for storing predictions and true values
    y_predict = []
    y_true = []
    
    # Initialize an table for comparing the performance of the models
    benchmark_df = pd.DataFrame(columns = [
        "$i$", 
        "$k_{i}$", 
        "$E^{\\text{test}}_{i}$ (KNN)", 
        "$\\lambda^{*}_{i}$", 
        "$E^{\\text{test}}_{i}$ (LR)", 
        "$E^{\\text{test}}_{i}$ (baseline)"
    ])
    
    # Cross validation
    print("RUNNING CROSS VALIDATION FOR CLASSIFICATION MODELS")    
    #-------------------------------------------------------------------------------------Entering OUTER FOLD
    # Outer loop index
    i = 0

    # Outer loop progress bar
    outer_progress = tqdm(total = K1, desc = "OUTER FOLDS", position=0)

    for outer_fold, (train_index_outer, test_index_outer) in enumerate(CV_outer_1.split(X), start = 1):
        # Update the outer loop progress description with current fold
        outer_progress.set_description(f"OUTER FOLD ({outer_fold}/{K1})")
        outer_progress.update(1)

        # Extract training and test set for current CV outer fold
        X_train_outer_1 = X[train_index_outer,:]
        y_train_outer_1 = y[train_index_outer]
        X_test_outer_1  = X[test_index_outer,:]
        y_test_outer_1  = y[test_index_outer]
        
        # Inner loop index
        j = 0
        #-------------------------------------------------------------------------------------Entering INNER FOLD
         # Inner loop progress bar (displayed below the outer bar)
        with tqdm(total = K2, desc = f"INNER FOLDS", position = 1, leave = False) as inner_progress:
            for inner_fold, (train_index_inner, test_index_inner) in enumerate(CV_inner_2.split(X_train_outer_1), start = 1):
                # Update the inner loop description with the current fold
                inner_progress.set_description(f"INNER FOLD ({inner_fold}/{K2})")
                inner_progress.update(1)

                # Extract training and test set for current CV inner fold
                X_train_inner_2 = X[train_index_inner,:]
                y_train_inner_2 = y[train_index_inner]
                X_test_inner_2  = X[train_index_inner,:]
                y_test_inner_2  = y[train_index_inner]
                
                # Logistical Regression
                for idx in range(0, len(lambdas)):
                    clf = LogisticRegression(
                        penalty = "l2",  
                        C = 1 / lambdas[idx], 
                        max_iter = 4000
                    )
                    clf.fit(X_train_inner_2, y_train_inner_2)
                    y_predict_lr_inner_2 = clf.predict(X_test_inner_2).T
                    error_lr_inner_2[j, idx] = np.sum(y_predict_lr_inner_2 != y_test_inner_2) / len(y_test_inner_2)
            
                # KNN
                for idx in range(1, L+1):
                    knclassifier = KNeighborsClassifier(n_neighbors = idx)
                    knclassifier.fit(X_train_inner_2, y_train_inner_2)
                    y_predict_knn_inner_2 = knclassifier.predict(X_test_inner_2)
                    error_knn_inner_2[j, idx-1] = np.sum(y_predict_knn_inner_2 != y_test_inner_2) / len(y_test_inner_2)
                    
                j+=1
        #-------------------------------------------------------------------------------------Exiting INNER FOLD

        ### Baseline - Classify all as the most common class (mode)
        baseline = mode(y_train_outer_1).mode
        y_predict_base = np.ones((y_test_outer_1.shape[0]), dtype = int) * baseline
        
        # Calculate the generalization error for baseline (E^{test}_{i})
        error_base[i] = np.sum(y_predict_base != y_test_outer_1) / len(y_test_outer_1)

        ### Logistical Regression
        error_lr_min[i] = np.min(error_lr_inner_2.mean(0))
        opt_lambda_idx = np.argmin(error_lr_inner_2.mean(0))
        
        # Store the optimal lambda for logistic regression (lambda^{*}_{i})
        opt_lambda[i] = lambdas[opt_lambda_idx]
        
        # Train the model with the optimal lambda
        clf = LogisticRegression(
            penalty = "l2",             
            C = 1 / lambdas[i],
            max_iter = 4000
        )
        clf.fit(X_train_outer_1, y_train_outer_1)
        y_predict_lr_outer_1 = clf.predict(X_test_outer_1).T

        # Calculate the generalization error for logistic regression (E^{test}_{i})
        error_lr_outer_1[i] = np.sum(y_predict_lr_outer_1 !=y_test_outer_1) / len(y_test_outer_1)
        
        # Update weights
        w_class_0 += clf.coef_[0]
        w_class_1 += clf.coef_[1]
        w_class_2 += clf.coef_[2]

        ### K-Nearest Neighbors (KNN)
        # Train the model with the optimal number of neighbors (k^{*}_{i})
        opt_k = np.argmin(error_knn_inner_2.mean(0)) + 1
        knclassifier = KNeighborsClassifier(n_neighbors = opt_k)
        knclassifier.fit(X_train_outer_1, y_train_outer_1)
        y_predict_knn_outer_1 = knclassifier.predict(X_test_outer_1)
        
        # Calculate the generalization error for KNN (E^{test}_{i})
        error_knn_outer_1[i] = np.sum(y_predict_knn_outer_1 != y_test_outer_1) / len(y_test_outer_1)
        
        ### Add the results for this fold to the DataFrame
        benchmark_df = benchmark_df._append({
            "$i$": i+1,
            "$k_{i}$": opt_k, 
            "$E^{\\text{test}}_{i}$ (KNN)": round(error_knn_outer_1[i]*100, 1), 
            "$\\lambda^{*}_{i}$": opt_lambda[i], 
            "$E^{\\text{test}}_{i}$ (LR)": round(error_lr_outer_1[i]*100, 1),
            "$E^{\\text{test}}_{i}$ (baseline)": round(error_base[i]*100, 1)
        }, ignore_index = True)
        
        # Storing predictions and true values
        dy = []
        dy.append(y_predict_base)
        dy.append(y_predict_knn_outer_1)
        dy.append(y_predict_lr_outer_1)
        dy = np.stack(dy, axis=1)
        y_predict.append(dy)
        y_true.append(y_test_outer_1)
        
        i+=1
    
    # Close the outer progress bar at the end
    outer_progress.close()
    #-------------------------------------------------------------------------------------Exiting OUTER FOLD

    # Convert the predictions and true values to numpy arrays
    y_true = np.concatenate(y_true)
    y_predict = np.concatenate(y_predict)    
    
    # Save the benchmark results to a csv file and a latex file
    benchmark_df["$i$"] = benchmark_df["$i$"].astype(int)
    benchmark_df["$k_{i}$"] = benchmark_df["$k_{i}$"].astype(int)
    save_df_to_csv(benchmark_df, "results", "benchmark.csv")
    save_df_to_latex(benchmark_df, "results", "benchmark.tex")
    
    # Create plots and save them to the plots folder
    create_knn_plot(L, error_knn_inner_2)
    save_plot_to_folder("plots", "knn_error.pdf")

    create_logistic_regression_plot(lambdas, error_lr_inner_2)
    save_plot_to_folder("plots", "logistic_Regression_error.pdf")

    create_boxplot_classification(error_base, error_knn_outer_1, error_lr_outer_1)
    save_plot_to_folder("plots", "boxplot_error_distribution.pdf")

    # Save the weights to a csv file and a latex file
    save_weights_to_file(w_class_0, w_class_1, w_class_2, K1, header, labels)

    ### STATISTICAL ANALYSIS ###
    t_knn_bas, p_knn_base, CI_l_knn_base, CI_u_knn_base = coorelated_t_test_cross_validation(error_knn_outer_1, error_base)
    t_knn_lr, p_knn_lr, CI_l_knn_lr, CI_u_knn_lr = coorelated_t_test_cross_validation(error_knn_outer_1, error_lr_outer_1)
    t_lr_base, p_lr_base, CI_l_lr_base, CI_u_lr_base = coorelated_t_test_cross_validation(error_base, error_lr_outer_1)
    
    stat_df = pd.DataFrame(columns = [
        "Model 1", 
        "Model 2", 
        "t-score", 
        "p-value", 
        "95%-CI (lower)", 
        "95%-CI (upper)"
    ]) 

    stat_df = stat_df._append({
        "Model 1": "KNN", 
        "Model 2": "Baseline",
        "t-score": t_knn_bas,
        "p-value": p_knn_base,
        "95%-CI (lower)": CI_l_knn_base,
        "95%-CI (upper)": CI_u_knn_base
    }, ignore_index = True)

    stat_df = stat_df._append({
        "Model 1": "KNN", 
        "Model 2": "Logistic Regression",
        "t-score": t_knn_lr,
        "p-value": p_knn_lr,
        "95%-CI (lower)": CI_l_knn_lr,
        "95%-CI (upper)": CI_u_knn_lr
    }, ignore_index = True)

    stat_df = stat_df._append({
        "Model 1": "Baseline", 
        "Model 2": "Logistic Regression",
        "t-score": t_lr_base,
        "p-value": p_lr_base,
        "95%-CI (lower)": CI_l_lr_base,
        "95%-CI (upper)": CI_u_lr_base
    }, ignore_index = True)

    # Save the statistical analysis to a csv file and a latex file
    save_df_to_csv(stat_df, "results", "statistical_analysis.csv")
    