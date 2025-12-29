import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd

def perform_pca():
    output_path_plots = "./plots/"
    os.makedirs(output_path_plots, exist_ok=True) 

    # read dataset from csv file
    df = pd.read_csv("./data/clean_dataset.csv")

    # define the numeric attributes
    attributes_names = df.columns
    numeric_attributes = attributes_names[1:8]

    # Extract continous data to matrix X (np.array)
    X = np.array(df.values[:,1:8],  dtype=float)

    # wrap labels for better visualization in the plot ticks
    def wrap_labels(labels, width=10):
        return [label.replace(" ", "\n") for label in labels]
    wrapped_labels = wrap_labels(numeric_attributes, width=6)

    # get...
    M = X.shape[1] # --> number of attributes / features
    N = X.shape[0] # --> number of entries / observation

    # normalize the data by removing mean 
    Y = (X - np.ones((N, 1)) * X.mean(axis=0)) / X.std(axis=0)

    # Perform PCA by computing SVD of Y
    U, S, Vh = svd(Y, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()

    # Plot variance explained
    variance_explained_plot = plt.figure(figsize=(10, 8))

    plt.plot(range(1, len(rho) + 1), rho, "x-")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
    plt.plot([0, 10], [0.9, 0.9], "k--")
    #plt.title("Variance explained by principal components")
    plt.xlabel("Principal component", fontsize=16)
    plt.ylabel("Variance explained", fontsize=16)
    plt.legend(["Individual", "Cumulative", "Threshold (90%)"])
    plt.grid()
    plt.xlim(0.8, 7.2)
    variance_explained_plot.savefig(output_path_plots + "variance_explained_plot.pdf", format="pdf", bbox_inches="tight")

    # PC1_contributes = [contribute / Vh[0].sum() for contribute in Vh[0]]
    PC1_contributes = Vh[0]
    # Create the histogram plot
    histogram_plot = plt.figure(figsize=(10, 6))
    bars = plt.bar(x=wrapped_labels, height=PC1_contributes)
    plt.yticks([])
    for bar in bars:
        yval = bar.get_height()  # Get the height of the bar
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{round(yval, 3)}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
    spines = histogram_plot.gca().spines  
    spines['top'].set_visible(False)       
    spines['right'].set_visible(False)     
    spines['left'].set_visible(False)      
    histogram_plot.savefig(output_path_plots + "PC1_contributes.pdf", format="pdf", bbox_inches="tight")

    for target in [0, 9]: # --> !!!SET MANUALLY!!!
        # Extract the target attribute
        y = df.values[:,target]
        if target == 0:
            class_labels = np.array(["F","M","I"])
        else: 
            class_labels = np.array(["11-29","9-10","0-8"])
        C = class_labels.shape[0] # --> number of different classes

        # format the target name in a better looking way for plotting
        formatted_target_name = attributes_names[target].lower().replace("_", " ")

        # create different directories for saving plots stratified on different targets
        os.makedirs(output_path_plots + attributes_names[target], exist_ok=True) 
        output_directory_of_target = output_path_plots + attributes_names[target] + "/"

        # Transpose Vh
        V = Vh.T

        # Project the original data onto principal component space
        Z = Y @ V

        # Indices of the principal components to be plotted
        i = 0
        j = 1

        # Plot PCA only with PC1
        pc1_plot = plt.figure(figsize=(10, 2))
        #plt.title(f"PC1 division of abalones based on {formatted_target_name}")
        for c in range(C):
            # select indices belonging to class c:
            class_mask = y == class_labels[c]
            plt.plot(Z[class_mask, i], np.ones(Z[class_mask, i].shape[0]) * -c * 0.5, "o")
        plt.legend(class_labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("PC{0}".format(i + 1))
        plt.ylim(-1.5, 0.5)
        plt.yticks([])
        pc1_plot.savefig(output_directory_of_target + f"pc1_plot_{attributes_names[target]}.pdf", format="pdf", bbox_inches="tight")


        # Plot PCA of the data with PC1 and PC2
        pc1_2_plot = plt.figure()
        #plt.title(f"PCA division of abalones based on {formatted_target_name}")
        # Z = array(Z)
        for c in class_labels:
            # select indices belonging to class c:
            class_mask = y == c
            plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.3, markersize=8)
        plt.legend(class_labels)
        plt.xlabel("PC{0}".format(i + 1))
        plt.ylabel("PC{0}".format(j + 1))
        plt.grid(alpha=0.5)

        pc1_2_plot.savefig(output_directory_of_target + f"pc1_2_plot_{attributes_names[target]}.pdf", format="pdf", bbox_inches="tight")

    print("PCA plots are available in /plots/")
    print("PCA data tables are available in /data/") 

if __name__ == "__main__":
    perform_pca()