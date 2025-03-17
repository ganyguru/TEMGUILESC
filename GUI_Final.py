#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageDraw, ImageFont, ImagePalette, TiffTags, ImageTk, ImageEnhance
import PIL.Image
import DM3lib as dm3
from scipy import fftpack
import numpy as np
import imageio
import ncempy.io as nio
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import special
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import glob, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from ncempy.io import dm
import shutil

from os import path
from Graph import  PlotGraph
from Heatmap import  PlotHeatmap
from Temporalmap import  PlotTemporalmap
from PDF_export import  Export_PDF
from FFT import create_fft
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage.transform import resize
from scipy import ndimage as nd
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from tkinter import filedialog


# In[2]:


IMG_WIDTH = 512
IMG_HEIGHT = 1024
IMG_CHANNELS = 3


inputs = tf.keras.layers.Input((IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))

nfilters=1024
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
c1 = tf.keras.layers.Conv2D(int(nfilters/8), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(int(nfilters/8), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(int(nfilters/4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(int(nfilters/4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(int(nfilters/2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(int(nfilters/2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(int(nfilters), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(int(nfilters), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(int(nfilters*2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(int(nfilters*2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(int(nfilters), (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(int(nfilters), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(int(nfilters), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(int(nfilters/2), (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(int(nfilters/2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(int(nfilters/2), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(int(nfilters/4), (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(int(nfilters/4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(int(nfilters/4), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(int(nfilters/8), (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(int(nfilters/8), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(int(nfilters/8), (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon()) 


model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])
model.optimizer.lr=0.0001
model.load_weights('FFTmodelv1.hdf5')


# In[3]:


# class Application(tk.Frame):
#     def __init__(self, master=None):
#         super().__init__(master)
#         self.master = master
#         self.pack()
#         self.create_widgets()

#     def create_widgets(self):
#         # Main Frame
#         self.main_frame = Frame(self)
#         self.main_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)

#         # Input Parameters Frame
#         self.input_frame = LabelFrame(self.main_frame, text="Input Parameters")
#         self.input_frame.pack(fill=X, pady=5)

#         Label(self.input_frame, text="Was the instrument calibrated before measurement?", font=("Arial", 10)).pack(anchor=W, padx=10, pady=2)
#         self.calib_var = StringVar(value="0")
#         Radiobutton(self.input_frame, text="No", variable=self.calib_var, value="0").pack(anchor=W, padx=20)
#         Radiobutton(self.input_frame, text="Yes", variable=self.calib_var, value="1").pack(anchor=W, padx=20)

#         Label(self.input_frame, text="Specify initial and final slices for processing? (Initial Slice | Final Slice)", font=("Arial", 10)).pack(anchor=W, padx=10, pady=2)
#         self.calib_init = Entry(self.input_frame, width=5)
#         self.calib_init.pack(side=LEFT, padx=10)
#         self.calib_init.insert(0, "0")
#         self.calib_end = Entry(self.input_frame, width=5)
#         self.calib_end.pack(side=LEFT)
#         self.calib_end.insert(0, "-1")
#         Label(self.input_frame, text="Default : All slices [0:-1]", font=("Arial", 6)).pack(anchor=W, padx=10)

#         Label(self.input_frame, text="Populate custom d-spacing database (Initial Value (Å) | Final Value (Å) | Interval (Å) )", font=("Arial", 10)).pack(anchor=W, padx=10, pady=2)
#         self.db_init = Entry(self.input_frame, width=5)
#         self.db_init.pack(side=LEFT, padx=10)
#         self.db_init.insert(0, "-1")
#         self.db_end = Entry(self.input_frame, width=5)
#         self.db_end.pack(side=LEFT)
#         self.db_end.insert(0, "-1")
#         self.db_int = Entry(self.input_frame, width=5)
#         self.db_int.pack(side=LEFT)
#         self.db_int.insert(0, "-1")
#         Label(self.input_frame, text="Default : database.csv (-1)", font=("Arial", 6)).pack(anchor=W, padx=10)

#         Label(self.input_frame, text="Specify PixelSize (nm/pixel) if the slices mentioned have different magnification", font=("Arial", 10)).pack(anchor=W, padx=10, pady=2)
#         self.px_size_manual = Entry(self.input_frame, width=5)
#         self.px_size_manual.pack(anchor=W, padx=10)
#         self.px_size_manual.insert(0, "-1")
#         Label(self.input_frame, text="Default : From metadata of file (-1)", font=("Arial", 6)).pack(anchor=W, padx=10)

#         Label(self.input_frame, text="Specify Detection sensitivity", font=("Arial", 10)).pack(anchor=W, padx=10, pady=2)
#         self.Sens_slider = Scale(self.input_frame, from_=10, to=90, orient=HORIZONTAL)
#         self.Sens_slider.pack(anchor=W, padx=10)
#         self.Sens_slider.set(50)
#         Label(self.input_frame, text="Default : 50% threshold", font=("Arial", 6)).pack(anchor=W, padx=10)

#         # Control Buttons Frame
#         self.control_frame = Frame(self.main_frame)
#         self.control_frame.pack(fill=X, pady=5)

#         self.start_button = Button(self.control_frame, text="Start Processing", command=self.start_processing)
#         self.start_button.pack(side=LEFT, padx=10, pady=5)

# #         self.results_button = Button(self.control_frame, text="Display Results", command=self.display_results)
# #         self.results_button.pack(side=LEFT, padx=10, pady=5)

#         self.quit_button = tk.Button(self.control_frame, text="QUIT", fg="red", command=self.master.destroy)
#         self.quit_button.pack(side=LEFT, padx=10, pady=5)

#         # Status Bar
#         self.status_bar = tk.Label(self.main_frame, text="Ready", bd=1, relief=SUNKEN, anchor=W)
#         self.status_bar.pack(fill=X, pady=5)

#         # Progress Bar
#         self.progress = Progressbar(self.main_frame, orient=HORIZONTAL, length=100, mode='determinate')
#         self.progress.pack(fill=X, pady=5)

#     def start_processing(self):
#         try:
#             calib_init_val = int(self.calib_init.get())
#             calib_end_val = int(self.calib_end.get())
#             db_end_val = float(self.db_end.get())
#             db_init_val = float(self.db_init.get())
#             db_int_val = float(self.db_int.get())
#             px_size_manual_val = float(self.px_size_manual.get())
#             threshold_val = float("{:.2f}".format(float(100 - float(self.Sens_slider.get())) / 100.0))
#             self.progress.start(10)
#             self.status_bar.config(text="Processing...")
#             batch_process(calib_init_val, calib_end_val, db_end_val, db_init_val, db_int_val, px_size_manual_val, threshold_val)
#             self.status_bar.config(text="Processing completed.")
#             self.progress.stop()
#         except Exception as e:
#             messagebox.showerror("Error", str(e))
#             self.progress.stop()
#             self.status_bar.config(text="Error")

#     def display_results(self):
#         newWindow = Toplevel(self)
#         newWindow.title("TEM Results Window")
#         newWindow.geometry("960x600")

#         Label(newWindow, text="TEM IMAGE OF THE SAMPLE", font=("Arial", 10)).pack(anchor=W, padx=10, pady=10)
#         self.canvas = Canvas(newWindow, width=400, height=400)
#         self.canvas.pack(anchor=W, padx=10)
#         img = Image.open("placeholder.jpg")
#         img = img.resize((400, 400))
#         self.photo = ImageTk.PhotoImage(img)
#         self.canvas.create_image(0, 0, anchor=NW, image=self.photo)

#         self.result_listbox = Listbox(newWindow, width=50, height=20)
#         self.result_listbox.pack(side=LEFT, padx=10)
#         for file in os.listdir("processed_files"):
#             self.result_listbox.insert(END, file)

#         self.result_listbox.bind('<<ListboxSelect>>', self.on_result_select)

#     def on_result_select(self, event):
#         selected = self.result_listbox.get(self.result_listbox.curselection())
#         img_path = f"processed_files/{selected}/tem/{selected}_TEM.png"
#         img = Image.open(img_path)
#         img = img.resize((400, 400))
#         self.photo = ImageTk.PhotoImage(img)
#         self.canvas.create_image(0, 0, anchor=NW, image=self.photo)

# root = tk.Tk()
# root.title('TEM Image Processing')
# app = Application(master=root)
# app.mainloop()


# In[4]:


#model.save_weights('fftmodel_full.hdf5')


# In[ ]:



Calib_Flag=0
Spectra_Flag=0
ListIndexFlag=0
Materials_List=""
ListFileName=""
#def f1=0(y_true, y_pred):
#    return 1
#def dice_coefficient(y_true, y_pred):
#    numerator = 2 * tf.reduce_sum(y_true * y_pred)
#    denominator = tf.reduce_sum(y_true + y_pred)
#    return numerator / (denominator + tf.keras.backend.epsilon()) 
#model = tf.keras.models.load_model('FFTModel', custom_objects = {"f1": f1, "dice_coefficient": dice_coefficient})
#contour_model = tf.keras.models.load_model('model2.h5', custom_objects = {"f1": f1, "dice_coefficient": dice_coefficient})

def run():
    os.system('python Batch.py')
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
    
    def printSomething(self,content):
        # if you want the button to disappear:
        # button.destroy() or button.pack_forget()
        label = Label(root, text= content)
        #this creates a new label to the GUI
        label.pack() 

    def SetCalibVal():
        global Calib_Flag
        if var.get():
            Calib_Flag = var.get()
                    
    def SetSpectraval():
        global Spectra_Flag
        if var.get():
            Spectra_Flag = var.get()
    
    def batch_process(self,calib_init_val,calib_end_val,db_end_val,db_init_val,db_int_val,px_size_manual,threshold_val):
        results = []
        with open("factor.csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                results.append(row)
        factor=np.array(results)
        
        if db_end_val==-1 or db_init_val==-1 or db_int_val==-1: 
            df = pd.read_csv('database.csv',header=None)
            df.columns = ["components", "phase", "dspacing", "ranking"]
        else:
            d = []
            i=1
            for p in np.arange(db_int_val, db_end_val+db_init_val, db_init_val):
            
                d.append({
                    'Component': 'Component',
                    'Phase': str(i),
                    'dspacing':  str(p),
                    'ranking': '1'
                })
                i=i+1
            df = pd.DataFrame()
            df = pd.DataFrame(d)
            df.columns = ["components", "phase", "dspacing", "ranking"]
            df.to_csv('temp_db.csv', header=False, index=False)
            df = pd.read_csv('temp_db.csv',header=None)
            
        print(df)
        
        if not os.path.exists('processed_files'):
            os.makedirs('processed_files')

        
        for file in os.listdir("put_your_data_here"):
            print(file+" Processing started...")
            
            
            
            if file.endswith(".dm3") or file.endswith(".dm4"):
                final_file = "processed_files/"+file.split(".")[0]+"_dm"
                while(path.exists(final_file)):
                    final_file = "processed_files/"+final_file.split("_dm3/")[1]+"_1"
                final_file = final_file+'/'
                current_file = "put_your_data_here/"+file
                dmfile = dm.fileDM(current_file)
                dm3objects=0
                if dmfile.thumbnail:
                    dm3objects = dmfile.numObjects-1
                else:
                    dm3objects = dmfile.numObjects
                final_file = "processed_files/"+file.split(".")[0]+"_dm"
                mother_folder = str(final_file)
                os.makedirs(final_file+'/heatmap_csv')
                os.makedirs(final_file+'/sgraph_csv')
                if calib_end_val==-1:
                    calib_end_val=dm3objects
                for dm_slice in range(calib_init_val,calib_end_val):
                    print("Slice"+str(dm_slice+1))
                    final_file = "processed_files/"+file.split(".")[0]+"_dm"
                    final_file = final_file+'/'
                    final_file = final_file+'slice_'+str(dm_slice+1)+'/'
                    #os.makedirs(final_file)
                    tem_folder =final_file+'tem' 
                    pxsize=0
                    result=dmfile.getDataset(dm_slice)
                    if result['pixelUnit'][0]=='A':
                        pxsize=result['pixelSize'][1]*0.1
                    elif result['pixelUnit'][0]=='nm':
                        pxsize=result['pixelSize'][1]
                        
                        
                    if px_size_manual!=-1:
                        pxsize=px_size_manual
                    os.makedirs(tem_folder)
                    temname = tem_folder+"/"+file.split(".")[0]+"_TEM.png"
                    plt.imsave(temname, result['data'], cmap='gray')
                    create_fft(file,factor,final_file,mother_folder,df,model,pxsize,'dm',threshold_val,dm_slice)
                PlotHeatmap(mother_folder)
                #Export_PDF(mother_folder)
                    
                    
            elif file.endswith(".mrc"):
                final_file = "processed_files/"+file.split(".")[0]+"_mrc"
                while(path.exists(final_file)):
                    final_file = "processed_files/"+final_file.split("_mrc/")[1]+"_1"
                final_file = final_file+'/'
                current_file = "put_your_data_here/"+file
                mrc0 = nio.mrc.mrcReader(current_file)
                final_file = "processed_files/"+file.split(".")[0]+"_mrc"
                mother_folder = str(final_file)
                os.makedirs(final_file+'/heatmap_csv')
                os.makedirs(final_file+'/sgraph_csv')
                if calib_end_val==-1:
                    calib_end_val=len(mrc0['data'])
                for mrc_slice in range(calib_init_val,calib_end_val):
                    print("Slice"+str(mrc_slice+1))
                    final_file = "processed_files/"+file.split(".")[0]+"_mrc"
                    final_file = final_file+'/'
                    final_file = final_file+'slice_'+str(mrc_slice+1)+'/'
                    #os.makedirs(final_file)
                    tem_folder =final_file+'tem' 
                    pxsize=0
                    if mrc0['pixelUnit']=='A':
                        pxsize=mrc0['pixelSize'][1]*0.1
                    elif mrc0['pixelUnit']=='nm':
                        pxsize=mrc0['pixelSize'][1]
                    os.makedirs(tem_folder)
                    if px_size_manual!=-1:
                        pxsize=px_size_manual
                    temname = tem_folder+"/"+file.split(".")[0]+"_TEM.png"
                    plt.imsave(temname, mrc0['data'][mrc_slice], cmap='gray')
                    create_fft(file,factor,final_file,mother_folder,df,model,pxsize,'mrc',threshold_val,mrc_slice)
                PlotHeatmap(mother_folder)
                #Export_PDF(mother_folder)
                
            elif file.endswith(".ser"):
                final_file = "processed_files/"+file.split(".")[0]+"ser"
                while(path.exists(final_file)):
                    final_file = "processed_files/"+final_file.split("_ser/")[1]+"_1"
                final_file = final_file+'/'
                current_file = "put_your_data_here/"+file
                ser0 = nio.ser.serReader(current_file)
                final_file = "processed_files/"+file.split(".")[0]+"_ser"
                mother_folder = str(final_file)
                os.makedirs(final_file+'/heatmap_csv')
                os.makedirs(final_file+'/sgraph_csv')
                data = ser0['data']
                data_shape = np.shape(ser0['data']) 
                if len(data.shape) == 2:
                    # Convert to (4096, 4096, 1)
                    data = data.reshape(data_shape[0], data_shape[1], 1) 
                if calib_end_val==-1:
                    calib_end_val=data.shape[2]
                for ser_slice in range(calib_init_val,calib_end_val):
                    print("Slice"+str(ser_slice+1))
                    final_file = "processed_files/"+file.split(".")[0]+"_ser"
                    final_file = final_file+'/'
                    final_file = final_file+'slice_'+str(ser_slice+1)+'/'
                    #os.makedirs(final_file)
                    tem_folder =final_file+'tem' 
                    pxsize=0
                    if ser0['pixelUnit'][0]=='A':
                        pxsize=ser0['pixelSize'][1]*0.1
                    elif ser0['pixelUnit'][0]=='nm':
                        pxsize=ser0['pixelSize'][1]
                    elif ser0['pixelUnit'][0]=='m':
                        pxsize=ser0['pixelSize'][1]*1e9
                    os.makedirs(tem_folder)
                    if px_size_manual!=-1:
                        pxsize=px_size_manual
                    temname = tem_folder+"/"+file.split(".")[0]+"_TEM.png"
                    plt.imsave(temname, data[:,:,ser_slice], cmap='gray')
                    create_fft(file,factor,final_file,mother_folder,df,model,pxsize,'ser',threshold_val,ser_slice)
                PlotHeatmap(mother_folder)
                    
                    
                    
            elif file.endswith(".emd"):
                final_file = "processed_files/"+file.split(".")[0]+"_emd"
                while(path.exists(final_file)):
                    final_file = "processed_files/"+final_file.split("_emd/")[1]+"_1"
                final_file = final_file+'/'
                current_file = "put_your_data_here/"+file
                with nio.emdVelox.fileEMDVelox(current_file) as emd1:
                    print(emd1) # print information about the file
                    im0, metadata0 = emd1.get_dataset(0)
                    if len(im0.shape)<3:
                        im0 = im0.reshape(im0.shape[0], im0.shape[1], 1)
                        size=1
                    else:
                        size =im0.shape[2]
                print("Shape of the image is" + str(im0.shape))
                final_file = "processed_files/"+file.split(".")[0]+"_emd"
                mother_folder = str(final_file)
                os.makedirs(final_file+'/heatmap_csv')
                os.makedirs(final_file+'/sgraph_csv')
                if calib_end_val==-1:
                    calib_end_val=size
                for emd_slice in range(calib_init_val,calib_end_val):
                    print("Slice"+str(emd_slice+1))
                    final_file = "processed_files/"+file.split(".")[0]+"_emd"
                    final_file = final_file+'/'
                    
                    final_file = final_file+'slice_'+str(emd_slice+1)+'/'
                    
                    #os.makedirs(final_file)
                    tem_folder =final_file+'tem' 
                    pxsize=0
                    if metadata0['pixelSizeUnit'][0]=='A':
                        pxsize=metadata0['pixelSize'][0]*0.1
                    elif metadata0['pixelSizeUnit'][0]=='nm':
                        pxsize=metadata0['pixelSize'][0]
                    os.makedirs(tem_folder)
                    if px_size_manual!=-1:
                        pxsize=px_size_manual
                    temname = tem_folder+"/"+file.split(".")[0]+"_TEM.png"
                    plt.imsave(temname, im0[:,:,emd_slice], cmap='gray')
                    
                    create_fft(file,factor,final_file,mother_folder,df,model,pxsize,'emd',threshold_val,emd_slice)
                PlotHeatmap(mother_folder)
                #Export_PDF(mother_folder)
                
            plt.cla()
            plt.clf()
            plt.close('all')
            
            
   

    def openFlagWindow(self):
        global labels 
        global recs
        # Toplevel object which will
        # be treated as a new window
        
        style = Style(root)
        style.theme_use('classic')
        style.configure('Test.TLabel', background= 'green',foreground='white')
        newWindow = Toplevel(self)
 
        # sets the title of the
        # Toplevel widget
        newWindow.title("TEM Results Window")
 
        # sets the geometry of toplevel
        newWindow.geometry("960x600")
        
        calib_val = StringVar(newWindow, "0")
 
        spectra_val = StringVar(newWindow, "0")
        # Dictionary to create multiple buttons 
        cvalues = {"No" : "0", 
                  "Yes" : "1"} 
        
        svalues = {"No" : "0", 
                  "Yes" : "1"} 

        
        #material_txt = Text(newWindow, height = 1, width = 20)
        #material_txt.place(x = 140,y = 120)
        # Loop is used to create multiple Radiobuttons 
        # rather than creating each button separately 
        calib_label = Label(newWindow, text = "Was the instrument calibrated before measurement?",font=("Arial", 10)).place(x = 140,y = 40)
        i=0
        
        validation_label = Label(newWindow, text = "",font=("Arial", 10))
        validation_label.place(x = 140,y = 220)   
        def validate():           
            global Materials_List
            
            calib_init_val = calib_init.get("1.0",END)
            calib_end_val = calib_end.get("1.0",END)
            
            db_end_val = db_end.get("1.0",END)
            db_init_val = db_init.get("1.0",END)
            db_int_val = db_int.get("1.0",END)
            px_size_manual_val=px_size_manual.get("1.0",END)
            threshold_val = float("{:.2f}".format(float(100-float(Sens_slider.get()))/100.0))
            
            
            
            #pattern = r'[A-Za-z.*\s.*,]'
#             if re.fullmatch(pattern, stripped_text) is None:
#                 print(stripped_text)    
#             else:
#                 #self.batch_process()
#                 print('Kundi')
            self.batch_process(int(calib_init_val),int(calib_end_val),float(db_end_val),float(db_init_val),float(db_int_val),float(px_size_manual_val),threshold_val)
                
        def show_values(new_value):
            slider_label.config(text = float("{:.1f}".format(float(new_value))))
                
        def create_textboxes():
            for i in range(3):
                textbox = tk.Entry(root, width=20)  # Adjust width as needed
                textbox.pack(side="left", padx=5)   # padx adds a bit of space between boxes
        
        def SetCalibval():
            global Calib_Flag
            if calib_val.get():
                Calib_Flag = calib_val.get()
                print(Calib_Flag)
                    
        def SetSpectraval():
            global Spectra_Flag
            if spectra_val.get():
                Spectra_Flag = spectra_val.get()
                print(Spectra_Flag)
                
        for (text, value) in cvalues.items(): 
            calib_radio = Radiobutton(newWindow, text = text, variable = calib_val, value = value, command=SetCalibval).place(x = 140,y = 60+i*20)
            i=i+1
        
        
        
        mat_txt_label = Label(newWindow, text = "Specify initial and final slices for processing ? (Initial Slice | Final Slice)",font=("Arial", 10)).place(x = 140,y = 100)  
        #calib_txt =[]
        calib_init= Text(newWindow, height = 1, width = 4)
        calib_init.place(x = 140,y = 120)
        calib_end= Text(newWindow, height = 1, width = 4)
        calib_end.place(x = 170,y = 120)
        
        calib_init.insert(END,0)
        calib_end.insert(END,-1)
        
        mat_txt_label = Label(newWindow, text = "Default : All slices [0:-1]",font=("Arial", 6)).place(x = 210,y = 120) 
        
        mat_txt_label = Label(newWindow, text = "Populate custom d-spacing database (Initial Value (Å) | Final Value (Å) | Interval (Å) )",font=("Arial", 10)).place(x = 140,y = 140)  
        #calib_txt =[]
        
        mat_txt_label = Label(newWindow, text = "Default : database.csv (-1)",font=("Arial", 6)).place(x = 240,y = 160) 
        db_init= Text(newWindow, height = 1, width = 4)
        db_init.place(x = 200,y = 160)
        db_end= Text(newWindow, height = 1, width = 4)
        db_end.place(x = 170,y = 160)
        db_int= Text(newWindow, height = 1, width = 4)
        db_int.place(x = 140,y = 160)
        
        mat_txt_label = Label(newWindow, text = "Specify PixelSize (nm/pixel) if the slices mentioned have different magnification",font=("Arial", 10)).place(x = 140,y = 180)  
        #calib_txt =[]
        px_size_manual= Text(newWindow, height = 1, width = 4)
        px_size_manual.place(x = 140,y = 200)
        px_size_manual.insert(END,-1)
        mat_txt_label = Label(newWindow, text = "Default : From metadata of file (-1)",font=("Arial", 6)).place(x = 172,y = 200) 
            
        db_init.insert(END,-1)
        db_end.insert(END,-1)
        db_int.insert(END,-1)
        mat_txt_label = Label(newWindow, text = "Specify Detection sensitivity",font=("Arial", 10)).place(x = 140,y = 220)  
        Sens_slider = Scale(newWindow, from_=10, to=90, orient=HORIZONTAL,command=show_values)
        Sens_slider.place(x = 140,y = 240)
        slider_label = Label(newWindow, text = "",font=("Arial", 6))
        slider_label.place(x = 250,y = 240) 
        Sens_slider.set(50)
        mat_txt_label = Label(newWindow, text = "Default : 50% threshold",font=("Arial", 6)).place(x = 295,y = 240) 
#         material_txt = Text(newWindow, height = 1, width = 20)
#         material_txt.place(x = 140,y = 120)
        
        
#         mat_txt_label = Label(newWindow, text = "What elements do you expect to observe? (Elements seperated by commas)",font=("Arial", 10)).place(x = 200,y = 100 , width =50)
        
#         i=0
        
#         spectra_label = Label(newWindow, text = "Were the elements present validated using spectroscopy?",font=("Arial", 10)).place(x = 140,y = 140)
#         spectra_radio = Radiobutton(newWindow, text = "No", variable = spectra_val, value = "0",command=SetSpectraval).place(x = 140,y = 160)
#         spectra_radio = Radiobutton(newWindow, text = "Yes", variable = spectra_val, value = "1",command=SetSpectraval).place(x = 140,y = 180)
       
       
        
        
        C_Processing = Button(newWindow, text="Continue Processing",command=validate)
        C_Processing.place(x = 140,y = 270)
       
                
       
            
        
        
        
        
        
        
    def openResWindow(self):
        global labels 
        global recs
        # Toplevel object which will
        # be treated as a new window
        
        style = Style(root)
        style.theme_use('classic')
        style.configure('Test.TLabel', background= 'green',foreground='white')
        newWindow = Toplevel(self)
 
        # sets the title of the
        # Toplevel widget
        newWindow.title("TEM Results Window")
 
        # sets the geometry of toplevel
        newWindow.geometry("960x600")
        bright_value = tk.DoubleVar()
        contrast_value = tk.DoubleVar()
        def get_bright_value():
            return '{: .2f}'.format(bright_value.get())
        def get_contrast_value():
            return '{: .2f}'.format(contrast_value.get())
       
        bright_slider = Scale(newWindow,from_=0,to=100,orient='horizontal', command=slider_changed,variable=current_value) 
        contrast_slider = Scale(newWindow,from_=0,to=100,orient='horizontal', command=slider_changed,variable=current_value)
        tem_image = Label(newWindow, text = "TEM IMAGE OF THE SAMPLE",font=("Arial", 10)).place(x = 140,y = 40)  
        sample_name = Label(newWindow, text = "SAMPLE NAME",font=("Arial", 10)).place(x = 750,y = 40)  
        comp_list = Label(newWindow, text = "COMPONENTS LIST",font=("Arial", 10)).place(x = 495,y = 150)  
        match_result = Label(newWindow, text="",font=("Arial", 10))
        match_result.place(x = 160,y = 500)
       
        
        canvas = Canvas(newWindow, width = 400, height = 400)
        canvas2 = Canvas(newWindow, width = 400, height = 200)      
        canvas.place(x=50, y=70)
        canvas2.place(x=500, y=400)
        # A Label widget to show in toplevel
        #Label(newWindow,text ="This is a new window").pack()
        #label1.grid(row=1,column=2)
        listbox = Listbox(newWindow, width=18,height =17,font=("Arial", 12),selectbackground='green',selectforeground='white', exportselection=False) 
        
        mat_box = Listbox(newWindow, width=14,height =8,font=("Arial", 12),selectbackground='green',selectforeground='white', exportselection=False) 
        #listbox.grid(row=1,column=3) 
        i =1
        for file in os.listdir("processed_files"):
            listbox.insert(i,file)  
            i = i+1
            
        listbox.place(x=700, y=70)
        mat_box.place(x=485, y=180)
        
        selected_df = ''
        selected_value=''
        labels = []
        recs=[]
        
        def toggle():
            index = int(mat_box.curselection()[0])
            value = mat_box.get(index)
            if toggle.config('text')[-1] == 'Contour_State-1':
                toggle.config(text='Contour_State-2')
                if(value!='View All' and value!='Other Regions'):
                    file_name = list(self.selected_df.loc[self.selected_df['material']==value]['file_name'])[0]
                    file='processed_dm3/'+self.selected_value+'/mask/masked/2/'+file_name
                elif(value=='Other Regions'):
                    file='processed_dm3/'+self.selected_value+'/mask/masked/2/fft_mask_area_other.png'
                else:
                    file='processed_dm3/'+self.selected_value+'/mask/masked/2/all_mask.png'
                #print(file)
                img = PIL.Image.open(file)  # PIL solution
                img = img.resize((400, 400))
                self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage   
            else:
                toggle.config(text='Contour_State-1')
                if(value!='View All' and value!='Other Regions'):
                    file_name = list(self.selected_df.loc[self.selected_df['material']==value]['file_name'])[0]
                    file='processed_dm3/'+self.selected_value+'/mask/masked/1/'+file_name
                elif(value=='Other Regions'):
                    file='processed_dm3/'+self.selected_value+'/mask/masked/1/fft_mask_area_other.png'
                else:
                    file='processed_dm3/'+self.selected_value+'/mask/masked/1/all_mask.PNG'
                img = PIL.Image.open(file)  # PIL solution
                img = img.resize((400, 400))
                self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage 
            canvas.create_image(0,0, anchor='nw', image=self.cimg) 
                
        #toggle = Button(newWindow, text="Contour_State-1",command=toggle)
        #toggle.place(x = 491,y = 100)  
        
        
        def my_mat(my_widget):
            #toggle.config(text='Contour_State-1')
            my_w = my_widget.widget
            index = int(my_w.curselection()[0])
            value = my_w.get(index)
            if(value!='View All'):
                match_result['text']='D-SPACING : '+ str(np.round(list(self.selected_df.loc[self.selected_df['material']==value]['value'])[0],2)) + 'Å'
                match_result['style']= 'Test.TLabel'
                file_name = list(self.selected_df.loc[self.selected_df['material']==value]['material'])[0]
                file='processed_files/'+ListFileName+'/'+self.selected_value+'/mask/'+file_name+'_img.png'
                img = PIL.Image.open(file)  # PIL solution
                img = img.resize((400, 400))
                self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage   
            else:
                file=glob.glob('processed_files/'+self.selected_value+'/mask/all_masks*.png')[0]
                img = PIL.Image.open(file)  # PIL solution
                img = img.resize((400, 400))
                self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage   
                match_result['text']=''
                match_result['style']=''
            
            canvas.create_image(0,0, anchor='nw', image=self.cimg) 
        
        def my_upd(my_widget):
            global labels
            global ListIndexFlag
            global ListFileName
            global recs
            my_w = my_widget.widget
            #toggle.config(text='Contour_State-1')
            index = int(my_w.curselection()[0])
            value = my_w.get(index)
            
            
            if(ListIndexFlag==0):
                my_w.delete(0,my_w.size())
                my_w.insert(1,'...')
                cnti=2
                ListFileName=value
                for file in os.listdir("processed_files/"+value):
                    my_w.insert(cnti,file)  
                    cnti = cnti+1
                ListIndexFlag=1
                return
            
            if(value=='...'):
                my_w.delete(0,my_w.size())
                cnti=1
                for file in os.listdir("processed_files"):
                    my_w.insert(cnti,file)  
                    cnti = cnti+1
                ListIndexFlag=0
                return
            
            
            file=glob.glob('processed_files/'+ListFileName+'/'+value+'/tem/*_TEM.png')[0]
            img = PIL.Image.open(file)  # PIL solution
            img = img.resize((400, 400))
            self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage   
            self.selected_value=value
            canvas.create_image(0,0, anchor='nw', image=self.cimg) 
            csv_file=glob.glob('processed_files/'+ListFileName+'/'+value+'/matdetails/mat_details*.csv')[0]
            match_result['text']=''
            match_result['style']=''
            self.selected_df = pd.read_csv(csv_file)
            i=1
            for k in labels:
                k.place_forget()
            for k in recs:
                canvas2.delete(k)
            labels=[]
            recs=[]
            canvas2.place_forget()
            canvas2.place(x=500, y=400)
            mat_box.delete(0, 'end')
            
            for x in list(self.selected_df['material']):
                mat_box.insert(i,x)
                if(i==1):
                    color_code = 'blue'
                elif(i==2):
                    color_code = 'red'
                elif(i==3):
                    color_code = 'green'
                elif(i==4):
                    color_code = 'purple'
                elif(i==4):
                    color_code = 'yellow'
                y0 = 0+((i-1)*30)
        
                
                myrectangle = canvas2.create_rectangle(0, y0, 20, y0+20, fill=color_code)
                recs.append(myrectangle)
                l = Label(canvas2,text = str(x),font=("Arial", 7))
                l.place(x = 30,y = 2+y0)
                #self.labels.append(l)
                i=i+1
                labels.append(l)
#             y0 = 0+((i-1)*30)
#             myrectangle = canvas2.create_rectangle(0, y0, 20, y0+20, fill='cyan')
#             recs.append(myrectangle)
#             l = Label(canvas2,text = 'Other Regions',font=("Arial", 7))
#             l.place(x = 30,y = 2+y0)
#             labels.append(l)       
            mat_box.insert(i,'View All')    
            
         
        listbox.bind('<<ListboxSelect>>', my_upd)
        mat_box.bind('<<ListboxSelect>>', my_mat)
        img = PIL.Image.open("placeholder.jpg")  # PIL solution
        img = img.resize((400, 400))
        self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage     
        canvas.create_image(0,0, anchor='nw', image=self.cimg) 
        
        
    def create_widgets(self):
        self.hi_there = tk.Button(self)
        
 
    # Setting icon of master window
        
        self.hi_there["text"] = "Start Processing"
        self.hi_there["command"] = self.openFlagWindow
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")
        self.res_window = tk.Button(self)
        self.res_window["text"] = "Display Results Window"
        self.res_window.pack(side="bottom")
        self.res_window["command"] = self.openResWindow
             
        #img = PhotoImage(file="ball.ppm")      
        #canvas.create_image(20,20, anchor=NW, image=img)
    def say_hi(self):
        print("hi there, everyone!")
        
        

root = tk.Tk()
root.tk.call('tk', 'scaling', 2.0)
root.title('TEM Image Processing')
#root.iconphoto(False, 'logo.png')
canvas = Canvas(root, width = 300, height = 300)      
canvas.pack()      
img = PhotoImage(master = canvas,file="logo.png")      
canvas.create_image(55,75, anchor=NW, image=img) 
app = Application(master=root)
 
app.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




