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
from skimage import measure, color, io

def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)

def PlotHeatmap(mother_folder):
    path = mother_folder+"/heatmap_csv"
  
    # csv files in the path
    files = glob.glob(path + "/*.csv")
  
    # defining an empty list to store 
    # content
    data_frame = pd.DataFrame()
    content = []
    df = pd.DataFrame()
    # checking all the csv files in the 
    # specified path
    for filename in files:
    
    # reading content of csv file
    # content.append(filename)
        df = pd.read_csv(filename, index_col=None)
        content.append(df)
      
    if df.empty:
        return;

    data_frame = pd.concat(content)
    a = pd.pivot_table(data_frame, index='frames', columns='material',values='intensity', fill_value=0)
    normalized_df=((a/a.max())*100)
    new_index = pd.Index(np.arange(min(normalized_df.index),max(normalized_df.index)+1,1), name="frames")
    normalized_df = normalized_df.set_index(normalized_df.index).reindex(new_index)
    normalized_df = normalized_df.fillna(0)
    columns = list(normalized_df.columns)
    
    



    fig=plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    """                                                                                                                                                    
    Scaling is done from here...                                                                                                                           
    """
    x_scale=4
    y_scale=2
    z_scale=1

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0
    colors = ['r', 'g', 'b']
    for i, (c, z) in enumerate(zip(columns, range(len(columns)))):
        xs = np.arange(min(normalized_df.index),max(normalized_df.index)+1,1)
        ys = list(normalized_df[columns[i]])
        #print(len(xs))
        #print(len(ys))
        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        #print(cs)
        ax.bar(xs, ys, z,zdir='x')



    ax.set(xticks=range(len(columns)), xticklabels=columns,zticks=np.arange(0, 101, 25))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.xticks(fontsize=18, rotation=90)
    plt.ylim(min(normalized_df.index), max(normalized_df.index)+1)
    ax.set_ylabel('Frames',fontsize=20,labelpad=15)
    ax.set_zlabel('Relative Intensity',fontsize=20,labelpad=15)


    plt.savefig(mother_folder+'/Heatmap.png')
    normalized_df.to_csv(mother_folder+'/Heatmap_Database.csv')
    plt.cla()
    plt.clf()
    plt.close('all')
    print("Processing done\n")