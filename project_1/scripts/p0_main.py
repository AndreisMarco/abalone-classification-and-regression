from p1_get_dataset import create_dataset
from p2_summary_statistics import generate_summary_stats
from p3_data_visualization import create_data_plots
from p4_PCA import perform_pca
import utils

def main():
    print("1) Manipulation of raw to create the dataset")
    create_dataset()

    print("2) Calculating summary statistics...")
    generate_summary_stats() 


    print("3) Creating visualization of data...")
    create_data_plots() 

    print("4) Performing PCA...")
    perform_pca()
    
    print("Done :D")

# Execute the main function
if __name__ == "__main__":
    main()