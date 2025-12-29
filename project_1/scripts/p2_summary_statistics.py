import pandas as pd
from utils import save_df_to_csv
import os

def generate_summary_stats():
    # read dataset from csv file
    df = pd.read_csv("./data/dataset.csv")

    # define the numeric attributes
    numeric_attributes = df.columns[1:9]

    # calculate basic statistics
    stats = {
        "summary_full": df.describe().drop("count"), # # Calculate summary statistics
        "summary_male": df[df["Sex"] == "M"].describe().drop("count"), # # Calculate summary statistics
        "summary_female": df[df["Sex"] == "F"].describe().drop("count"), # # Calculate summary statistics
        "summary_infant": df[df["Sex"] == "I"].describe().drop("count"), # # Calculate summary statistics
        "covariance": round(df[numeric_attributes].cov(),4), # calculate covariance matrix
        "correlation": round(df[numeric_attributes].corr(),4), # calculate correlation matrix
    }

    # save stastistics dataframe to csv file
    stats_path = "./data/stats.csv"

    # if file already exist remove it as to not create duplicates 
    if os.path.exists(stats_path):
        os.remove(stats_path)

    for df_name in stats.keys():
        save_df_to_csv(stats[df_name], path=stats_path, index_label=df_name)
    
    print("Summary statistics created and available in /data/stats.csv") 

    # create and save a dataset without outliers
    print(df[df["Height"] > 100])
    cleaned_df = df.drop(df[df["Height"] > 100].index)
    cleaned_df = cleaned_df.drop(cleaned_df[cleaned_df["Height"] == 0].index)
    cleaned_df.to_csv("./data/clean_dataset.csv", index=False)

if __name__ == "__main__":
    generate_summary_stats()


