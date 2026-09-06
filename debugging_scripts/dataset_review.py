import pandas as pd
import numpy as np
from config import GLOBAL_CONFIG





def analyze_dataset():
    DATASET_PATH = GLOBAL_CONFIG.pipeline.csv_path
    DATASET_PATH_DOWNSAMPLES = GLOBAL_CONFIG.pipeline.path_to_downsampled_dataset

    dataset = pd.read_csv(DATASET_PATH_DOWNSAMPLES)
    total_rows = len(dataset)

    print(f"Total number of rows: {total_rows}")

    # Check how many unvalid data is inside
    nan_count = dataset.isna().sum()

    print(f"Total number of unvalid data: {nan_count}")

    print(f"Columns name: {list(dataset.columns)}")

    # Now we check the amount of data: (we are at the moment focusing only on steering)

    straight_drive = (dataset['lane_offset'].abs() <= 0.05).sum()
    print(f"straight: {straight_drive}")
    # REturn 80+% this is a problem
    print((100/total_rows)*straight_drive)

    mask_deadzone = (dataset['lane_offset'].abs() <= 0.05) & (dataset['heading_error'].abs() <= 0.01)
    mask_extreme = (dataset['lane_offset'].abs() > 0.30) | (dataset['heading_error'].abs() > 0.05)
    mask_medium = ~mask_deadzone & ~mask_extreme

    count_A = mask_deadzone.sum()
    count_B = mask_medium.sum()
    count_C = mask_extreme.sum()
    print("Deadzone")
    print((100/total_rows)*count_A)
    print(count_A)
    print("Medium")
    print((100/total_rows)*count_B)
    print(count_B)
    print("Extreme")
    print((100/total_rows)*count_C)
    print(count_C)

#Deadzone
#80.37155080213903
#Extreme
#16.965508021390374
#Medium
#2.662941176470588

# Here we see a problem. The deadzone so basically dribing in a straight line is having 80% of complete dataset data. 
# This could potentionally lead to a problem we are having at the moment.
# Next step is to smartly downsample and maybe upsample(add some data for medium) and try to retrain the model


if __name__ == "__main__":
    analyze_dataset()
