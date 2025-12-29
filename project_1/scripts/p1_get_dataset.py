import pandas as pd 
import numpy as np
import os

def create_dataset():
    # get raw dataset
    original_data = pd.read_csv("./raw/abalone.data")

    # define the names of the attributes 
    attribute_names = ['Sex', 'Length', 'Diameter', 'Height',	'Whole weight',	'Shucked weight', 'Viscera weight',	'Shell weight', 'Rings']

    # extract raw values from the original dataframe
    raw_data = original_data.values

    # create dataframe rescaling the continuous attributes
    rescaled_data = raw_data[:,1:8].astype(float) * 200    
    df = pd.DataFrame(rescaled_data, columns=attribute_names[1:8])
    df = df.round(decimals=3)
    df.insert(0, attribute_names[0], raw_data[:,0])
    df.insert(8, attribute_names[8], raw_data[:,8])

    # create age_group attribute based on Waugh system
    bins = [0, 8, 10, 30]  # Define the edges of the bins
    labels = ["0-8", "9-10", "11-29"]
    df['Number_of_rings'] = pd.cut(df['Rings'], bins=bins, labels=labels, right=True)

    # Save the new df in the output folder
    os.makedirs("./data", exist_ok=True) 
    df.to_csv("./data/dataset.csv", index=False)

    print("Dataset created and available in /data/dataset.csv") 

if __name__ == "__main__":
    create_dataset()