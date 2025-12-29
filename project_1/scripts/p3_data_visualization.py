import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

def create_data_plots():
    output_path = "./plots/"
    os.makedirs(output_path, exist_ok=True) 

    # read dataset from csv file
    df = pd.read_csv("./data/dataset.csv")

    # define the numeric attributes
    attributes_names = df.columns
    numeric_attributes = attributes_names[1:8]

    X = df.values[:,1:8]
    M = len(numeric_attributes)
    N = len(df[numeric_attributes])

    # wrap labels for better visualization in the plot ticks
    def wrap_labels(labels, width=10):
        return [label.replace(" ", "\n") for label in labels]
    wrapped_labels = wrap_labels(numeric_attributes, width=6)

    # Plot boxplot of the distribution of numerical variables
    boxplot = plt.figure(figsize=(10, 6))  
    df[numeric_attributes].boxplot() 
    #plt.title('Distribution of physical measurements')
    plt.grid(alpha = 0.5)  
    plt.tight_layout()
    boxplot.savefig(output_path + "boxplots.pdf", format="pdf", bbox_inches="tight")

    # Plot violinplot of the distribution of numerical variables
    violin_plot = plt.figure(figsize=(14, 10)) 
    plt.violinplot(df[numeric_attributes], showmedians=True, )
    #plt.title('Distribution of physical measurements')
    plt.xticks(range(1,M+1), wrapped_labels, fontsize = 17)
    plt.yticks(fontsize = 17)
    plt.grid(alpha = 0.5)  
    plt.tight_layout()
    violin_plot.savefig(output_path + "violin_plots.pdf", format="pdf", bbox_inches="tight")

    # Plot histograms of the numerical variables
    histogram_plot = plt.figure(figsize=(10, 6))
    #plt.suptitle("Frequency distribution of physical measurements")
    u = np.floor(np.sqrt(M))
    v = np.ceil(float(M) / u)
    for i in range(M):
        plt.subplot(int(u), int(v), i + 1)
        plt.hist(X[:, i], color=(0.5, 0.5, 0.5), bins=15)
        plt.xlabel(numeric_attributes[i])
        plt.ylim(0, 2500)
        plt.grid(alpha = 0.5)  
    plt.tight_layout()
    histogram_plot.savefig(output_path + "histogram_plot.pdf", format="pdf", bbox_inches="tight")

    #tot weight vs. length semilog scale --> Potential target for linear regression
    weight_length = plt.figure(figsize=(10, 6))
    #plt.title("Linear relationship between abalones' Length and Whole weight")
    plt.scatter(df[numeric_attributes[0]], df[numeric_attributes[3]])
    plt.xlabel(numeric_attributes[0] + " (log10)", fontsize=16)
    plt.ylabel(numeric_attributes[3] + " (log10)", fontsize=16)   
    plt.yscale('log')
    plt.xscale('log')
    weight_length.savefig(output_path + "weight_length_plot.pdf")

    #length vs. diameter --> Potential target for linear regression
    length_diameter_plot = plt.figure(figsize=(10, 6))
    #plt.title("Linear relationship between abalones' Length and Diameter")
    plt.scatter(df[numeric_attributes[0]], df[numeric_attributes[1]])
    plt.xlabel(numeric_attributes[0])
    plt.ylabel(numeric_attributes[1])
    length_diameter_plot.savefig(output_path + "length_diameter_plot.pdf", format="pdf", bbox_inches="tight")

    #Scatter of all attributes one over the others and stratify on 
    relation_matrix_plot = plt.figure(figsize=(12, 10))
    #plt.suptitle("Matrix representation of 1:1 relations between each attribute")
    for m1 in range(M):
        for m2 in range(M):
            plt.subplot(M, M, m1*M + m2+1)
            plt.plot(np.array(X[:,m2]), np.array(X[:,m1]), ".", alpha=0.5)
            plt.xticks([])
            plt.yticks([])
            if m1 == M-1:
                plt.xlabel(numeric_attributes[m2], fontsize=11)
            if m2 == 0:
                plt.ylabel(numeric_attributes[m1], rotation=90, fontsize=11)                  
    plt.tight_layout()
    relation_matrix_plot.savefig(output_path + "relation_matrix_plot.pdf", format="pdf", bbox_inches="tight")

    # cycle on potential targets attributes for the classification task 
    for target in [0, 9]: # --> !!!SET MANUALLY!!!
        y = df.values[:,target]
        if target == 0:
            class_labels = np.array(["F","M","I"])
        else: 
            class_labels = np.array(["0-8","9-10","11-29"])
        C = class_labels.shape[0]

        formatted_target_name = attributes_names[target].lower().replace("_", " ")

        # create different directories for saving plots stratified on different targets
        os.makedirs(output_path + attributes_names[target], exist_ok=True) 
        output_directory_of_target = output_path + attributes_names[target] + "/"


        # Plot violinplot of the distribution of numerical variables divided by the target
        violin_plot = plt.figure(figsize=(22, 8))
        #plt.suptitle(f"Distribution of physical measurements stratified on {formatted_target_name}")
        for c in range(C):
            plt.subplot(1, C, c + 1)
            class_mask = y == class_labels[c]  
            plt.violinplot(X[class_mask].astype(float), showmedians=True)
            plt.title("Class: " + class_labels[c])
            plt.grid(alpha = 0.5)  
            plt.xticks(range(1, M + 1), [a for a in wrapped_labels], fontsize=14, rotation=30)
            plt.yticks(fontsize=10)
            plt.ylim(-10, 600)
            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.1)
        violin_plot.savefig(output_directory_of_target + f"stratified_violin_plot_{attributes_names[target]}.pdf", format="pdf", bbox_inches="tight")

        # Plot boxplot of the distribution of numerical variables divided by the target
        boxplot = plt.figure(figsize=(14, 7))
        #plt.suptitle(f"Distribution of physical measurements stratified on {formatted_target_name}")
        for c in range(C):
            plt.subplot(1, C, c + 1)
            class_mask = y == class_labels[c]  
            tmp = plt.boxplot(X[class_mask])
            plt.title("Class: " + class_labels[c])
            plt.grid(alpha = 0.5)  
            plt.xticks(range(1, M + 1), [a for a in wrapped_labels], fontsize=8, rotation=30) 
            plt.ylim(-10, 600)
        boxplot.savefig(output_directory_of_target + f"stratified_boxplot_{attributes_names[target]}.pdf", format="pdf", bbox_inches="tight")

    print("Data plots are available in /plots/") 
        
if __name__ == "__main__":
    create_data_plots()

