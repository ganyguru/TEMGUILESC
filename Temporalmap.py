from __main__ import *
from scipy.signal import find_peaks
import scipy.fftpack as fp
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob
import re
from skimage import measure, color, io

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

def PlotTemporalmap(mother_folder):
    path = mother_folder+"/sgraph_csv"
  
    # csv files in the path
    files = glob.glob(path + "/*.csv")
  
    # defining an empty list to store 
    # content
    # Create an empty list to store dataframes
    dfs = []

    # Loop through each CSV file
    for file in files:
        # Extract the slice number from the filename (assuming a pattern like slice_1.csv, slice_2.csv)
        slice_number = int(re.search(r'slice_(\d+)', file).group(1))  # Extract the number after "slice_"
    
        # Read the CSV file into a pandas DataFrame, specifying no header and column names
        df = pd.read_csv(file, header=None, names=['Diffraction Length', 'Intensity'])
        
        # Filter out rows where 'Diffraction Length' is greater than 5
        df = df[df['Diffraction Length'] <= 5]
    
        # Format 'Diffraction Length' to have 2 decimal places
        #df['Diffraction Length'] = df['Diffraction Length'].apply(lambda x: f"{x:.2f}")
    
        # Replace NaN values in 'Intensity' column with 0
        df['Intensity'] = df['Intensity'].fillna(0)
    
        # Add a 'Slice' column to the DataFrame
        df['Slice'] = slice_number
        
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    print(combined_df)
    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = combined_df.pivot(index='Diffraction Length', columns='Slice', values='Intensity')  # Corrected line

    # Create the heatmap using seaborn
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(heatmap_data, cmap='viridis')
    # Format y-axis labels to display 2 decimal places
    ax.set_yticklabels([f"{float(label.get_text()):.2f}" for label in ax.get_yticklabels()])
    ax.invert_yaxis()
    plt.title('Temporal Map of Diffraction Length vs Intensity')
    plt.xlabel('Slice')  # Corrected label
    plt.ylabel('Diffraction Length')  # Corrected label

    plt.savefig(mother_folder+'/Temporal_map.png')
    heatmap_data.to_csv(mother_folder+'/Temporal_Database.csv')
    plt.cla()
    plt.clf()
    plt.close('all')
    print("Processing done\n")