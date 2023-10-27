import PIL
from PIL import Image, ImageDraw, ImageFont, ImagePalette, TiffTags, ImageTk, ImageEnhance
import PIL.Image
import DM3lib as dm3
import scipy
from scipy import fftpack
import numpy as np
import imageio
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
import shutil
from os import path
import Graph
import FFT
import sys
from Graph import PlotGraph
from FFT import create_fft
import skimage
from skimage.transform import resize
from scipy import ndimage as nd
import pandas as pd
import tkinter
from tkinter import *
from tkinter.ttk import *

IMG_WIDTH = 512
IMG_HEIGHT = 1024
IMG_CHANNELS = 3

def resourcePath(relativePath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        basePath = sys._MEIPASS
    except Exception:
        basePath = os.path.abspath(".")

    return os.path.join(basePath, relativePath)


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

model.load_weights(resourcePath('FFTmodelv1.hdf5'))

import tkinter as tk
import os
import sys
#def f1(y_true, y_pred):
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


    
    def batch_process(self):
        results = []
        with open(resourcePath("factor.csv")) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                results.append(row)
        factor=np.array(results)
        
        df = pd.read_csv(resourcePath('database.csv'),header=None)

        if not os.path.exists('processed_dm3'):
            os.makedirs('processed_dm3')


        for file in os.listdir(resourcePath("dm3")):
            if file.endswith(".dm3") or file.endswith(".dm4"):
                print(file+" Processing started...")
                final_file = "processed_dm3/"+file.split(".")[0]
                while(path.exists(final_file)):
                    final_file = "processed_dm3/"+final_file.split("_dm3/")[1]+"_1"
                final_file = final_file+'/'
                os.makedirs(final_file)
                print(file.split('\\')[-1].split(".")[0])
                create_fft(resourcePath(f"dm3\{file}"),factor,final_file,df,model)
                current_file = "dm3/"+file
                os.makedirs(final_file+'DM3_file/')
                #shutil.move(current_file,final_file+'DM3_file/')
            plt.cla()
            plt.clf()
            plt.close('all')

            
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
        
        mat_box = Listbox(newWindow, width=14,height =6,font=("Arial", 12),selectbackground='green',selectforeground='white', exportselection=False) 
        #listbox.grid(row=1,column=3) 
        i =1
        for file in os.listdir("processed_dm3"):
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
                file='processed_dm3/'+self.selected_value+'/mask/'+file_name+'_img.png'
                img = PIL.Image.open(file)  # PIL solution
                img = img.resize((400, 400))
                self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage   
            else:
                file='processed_dm3/'+self.selected_value+'/mask/all_masks_'+self.selected_value+'.png'
                img = PIL.Image.open(file)  # PIL solution
                img = img.resize((400, 400))
                self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage   
                match_result['text']=''
                match_result['style']=''
            
            canvas.create_image(0,0, anchor='nw', image=self.cimg) 
        
        def my_upd(my_widget):
            global labels
            
            global recs
            my_w = my_widget.widget
            #toggle.config(text='Contour_State-1')
            index = int(my_w.curselection()[0])
            value = my_w.get(index)
            file='processed_dm3/'+value+'/tem/'+value+'_TEM.png'
            img = PIL.Image.open(file)  # PIL solution
            img = img.resize((400, 400))
            self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage   
            self.selected_value=value
            canvas.create_image(0,0, anchor='nw', image=self.cimg) 
            csv_file='processed_dm3/'+value+'/matdetails/mat_details'+value+'.csv'
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
        img = PIL.Image.open(resourcePath("placeholder.jpg"))  # PIL solution
        img = img.resize((400, 400))
        self.cimg = ImageTk.PhotoImage(img) # convert to PhotoImage     
        canvas.create_image(0,0, anchor='nw', image=self.cimg) 
        
        
    def create_widgets(self):
        self.hi_there = tk.Button(self)
        
 
    # Setting icon of master window
        
        self.hi_there["text"] = "Start Processing"
        self.hi_there["command"] = self.batch_process
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
img = PhotoImage(master = canvas,file=resourcePath("logo.png"))
canvas.create_image(55,75, anchor=NW, image=img) 
app = Application(master=root)
 
app.mainloop()


