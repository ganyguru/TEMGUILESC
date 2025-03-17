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

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Table, TableStyle, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd
import os


def Export_PDF(mother_folder):
    pdf_filename = 'Analysis_Results.pdf'
    pdf_path = mother_folder + '/' + pdf_filename
    print(pdf_path)
    if not os.path.exists(pdf_filename):
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []
    else:
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = doc.build(doc.getStory())  # Append to existing PDF

    
    #Get Slice folders    
    slice_folders = []
    for item in os.listdir(mother_folder):
        item_path = os.path.join(mother_folder, item)
        if os.path.isdir(item_path) and item.startswith('slice'):
            slice_folders.append(item)
            
    for slice_f in slice_folders:
           # Add Header (if provided)
        
#         styles = getSampleStyleSheet()
#         story.append(Paragraph(header_text, styles['Heading2']))
#         story.append(Spacer(1, 12))
        

        
        #Add Images
        folder_path = 'tem'  # Replace with the actual path to your 'fft' folder

         # Get a list of all files in the folder
        files = os.listdir(mother_folder+"/"+slice_f+"/"+folder_path)
        final_folder_path = mother_folder+"/"+slice_f+"/"+folder_path
        # Filter for PNG images
        png_images = [file for file in files if file.endswith('.png')]
        img = Image(final_folder_path+'/'+png_images[0])
        img.drawHeight = 200  # Adjust height as needed
        img.drawWidth = 200   # Adjust width as needed
        story.append(img)
        
        
        #Add Images
        folder_path = 'fft'  # Replace with the actual path to your 'fft' folder

        # Get a list of all files in the folder
        files = os.listdir(mother_folder+"/"+slice_f+"/"+folder_path)
        final_folder_path = mother_folder+"/"+slice_f+"/"+folder_path
        # Filter for PNG images
        png_images = [file for file in files if file.endswith('.png')]
        img = Image(final_folder_path+'/'+png_images[0])
        img.drawHeight = 200  # Adjust height as needed
        img.drawWidth = 200   # Adjust width as needed
        story.append(img)
        
        #Add Images
        folder_path = 'model' #place with the actual path to your 'fft' folder

         # Get a list of all files in the folder
        files = os.listdir(mother_folder+"/"+slice_f+"/"+folder_path)
        final_folder_path = mother_folder+"/"+slice_f+"/"+folder_path
        # Filter for PNG images
        png_images = [file for file in files if file.endswith('.png')]
        img = Image(final_folder_path+'/'+png_images[0])
        img.drawHeight = 200  # Adjust height as needed
        img.drawWidth = 200   # Adjust width as needed
        story.append(img)
        
        #Add Images
        folder_path = 'finalfft' #place with the actual path to your 'fft' folder

         # Get a list of all files in the folder
        files = os.listdir(mother_folder+"/"+slice_f+"/"+folder_path)
        final_folder_path = mother_folder+"/"+slice_f+"/"+folder_path
        # Filter for PNG images
        png_images = [file for file in files if file.endswith('.png')]
        img = Image(final_folder_path+'/'+png_images[0])
        img.drawHeight = 200  # Adjust height as needed
        img.drawWidth = 200   # Adjust width as needed
        story.append(img)
        
        story.append(PageBreak())

#     # Second Page: Table + 2 Images
#     if csv_data is not None and len(images) >= 6:
#         df = pd.read_csv(csv_data)
#         table_data = [df.columns.tolist()] + df.values.tolist()
#         table = Table(table_data)
#         table.setStyle(TableStyle([
#             ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#             ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),   

#             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#             ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
#             ('GRID', (0, 0), (-1, -1), 1, colors.black)   

#         ]))
#         story.append(table)
#         story.append(Spacer(1,   
#  12))

#         for i in range(4, 6):
#             img = Image(images[i])
#             img.drawHeight = 200
#             img.drawWidth = 200
#             story.append(img)
#         story.append(PageBreak())

#     # Third Page: Remaining Images
#     if len(images) > 6:
#         for i in range(6, len(images)):
#             img = Image(images[i])
#             img.drawHeight = 200
#             img.drawWidth = 200
#             story.append(img)

#     # Add Footer (if provided)
#     if footer_text:
#         story.append(Spacer(1, 12))
#         story.append(Paragraph(footer_text, styles['Normal']))

    doc.build(story)
    print(f"PDF updated for TEM image: {tem_image_name}")
