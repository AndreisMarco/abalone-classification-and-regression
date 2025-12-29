import pandas as pd
import os

def save_df_to_csv(df: pd.DataFrame, path: str, index_label:str=""):
    '''
    This function saves a pandas DataFrame to the provided CSV file path.
    If the file already exists, it appends the DataFrame.
    If the file doesn't exist, it creates a new CSV file with the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    path (str): The file path to save the CSV.
    index_label(str): The name of the index column.
    '''
    try:
        # Check if the file exists
        if not os.path.exists(path):
            # If it doesn't exist, create a new file as save there th df
            df.to_csv(path, index_label=index_label)
        else:
            # If it exists, append to the file an empty row ...
            with open(path, 'a') as f:
                f.write("\n")
            # ...and the df
            df.to_csv(path, mode='a', index_label=index_label)
    except Exception:
        print(f"Error occurred while saving the file to {path}")