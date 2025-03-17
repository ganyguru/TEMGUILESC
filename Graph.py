from __main__ import *
from scipy.signal import find_peaks
import scipy.fftpack as fp
import tensorflow as tf
import pandas as pd
import Heatmap
import PDF_export
from Heatmap import PlotHeatmap
import Temporalmap
from Temporalmap import  PlotTemporalmap
from PDF_export import  Export_PDF
from skimage import measure, color, io


def extract_component_features(markers, dfp, component_name):
    """
    Reconstruct segmented image containing only features of a specific component.
    
    Parameters:
    -----------
    markers : ndarray
        The labeled image from cv2.connectedComponents
    dfp : pandas.DataFrame
        DataFrame containing the component information for each feature
    component_name : str
        The name of the component to extract
        
    Returns:
    --------
    component_mask : ndarray
        Binary mask containing only features of the specified component
    """
    # Create an empty mask with the same shape as the markers image
    component_mask = np.zeros_like(markers, dtype=np.uint8)
    
    # Get the labels of features that belong to component X
    component_labels = dfp[dfp['c_phase'] == component_name]['label'].values
    
    # For each label in the component, add it to the mask
    for label in component_labels:
        component_mask[markers == label] = 255
    
    return component_mask


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)
    parts = 1
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    tbin=tbin[0:center[0]]
    tbin_r = tbin[0::parts]
    for i in range(1,parts):
        tbin_r=tbin_r + tbin[i::parts]
    #tbin_r = tbin[0::16]+tbin[1::16]+tbin[2::16]+tbin[3::16]+tbin[4::16]+tbin[5::16]+tbin[6::16]+tbin[7::16]
    radialprofile = tbin_r
    return radialprofile 

def radial_profile_avg(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)
    parts=1
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    tbin=tbin[0:center[0]]
    tbin_r = tbin[0::parts]
    for i in range(1,parts):
        tbin_r=tbin_r + tbin[i::parts]
    #tbin_r = tbin[0::8]+tbin[1::8]+tbin[2::8]+tbin[3::8]+tbin[4::8]+tbin[5::8]+tbin[6::8]+tbin[7::8]
    nr=nr[0:center[0]]
    nr_r = nr[0::parts]
    for i in range(1,parts):
        nr_r=nr_r + nr[i::parts]
    #nr_r = nr[0::8]+nr[1::8]+nr[2::8]+nr[3::8]+nr[4::8]+nr[5::8]+nr[6::8]+nr[7::8]
    radialprofile = tbin_r/nr_r
    return radialprofile  

def r2d(radius,pxsize,r,divs):
    d = radius/10
    return 1/((d*((r+1)/2))/(pxsize*r*2*divs))
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def getcomponent(y,df):
    x=float(y)*1
    
    #component = str(list(df.loc[df[2]==find_nearest(np.array(df[2]),x)][0])[0]) +' '+ str(list(df.loc[df[2]==find_nearest(np.array(df[2]),x)][1])[0])
    component = str(list(df[df['dspacing']==find_nearest(np.array(df['dspacing']),x)]['components'])[0])
    return component

def getphase(y,df):
    x=float(y)*1
    #component = str(list(df.loc[df[2]==find_nearest(np.array(df[2]),x)][0])[0]) +' '+ str(list(df.loc[df[2]==find_nearest(np.array(df[2]),x)][1])[0])
    phase = str(list(df.loc[df['dspacing']==find_nearest(np.array(df['dspacing']),x)]['phase'])[0])
    return phase

def getranking(y,df):
    x=float(y)*1
    #component = str(list(df.loc[df[2]==find_nearest(np.array(df[2]),x)][0])[0]) +' '+ str(list(df.loc[df[2]==find_nearest(np.array(df[2]),x)][1])[0])
    ranking = str(list(df.loc[df['dspacing']==find_nearest(np.array(df['dspacing']),x)]['ranking'])[0])
    return ranking

def reassign_components(df, dfp_avg):
    """
    Reassigns components in dfp_avg based on rankings and dspacing in df.

    Args:
      df: DataFrame with component, phase, dspacing, and ranking.
      dfp_avg: DataFrame with component, phase, dspacing, and ranking to be reassigned.

    Returns:
      Reassigned dfp_avg DataFrame.
    """

    df_sorted = df.sort_values('dspacing', ascending=False)
    
    # Keep track of changes in each iteration
    changes_made = True
    while changes_made:
        changes_made = False

        for index, row in dfp_avg.iterrows():
            component = row['components']
            phase = row['phase']
            ranking = row['ranking']

            # Find matching rows in df
            matching_rows = df_sorted[(df_sorted['components'] == component) & (df_sorted['phase'] == phase)]

            # Check if a preceding ranking exists
            preceding_ranking_exists = any(matching_rows['ranking'] < ranking)

            if not preceding_ranking_exists:
                # Exclude rows with same component, phase, and ranking
                df_filtered = df_sorted.drop(matching_rows[matching_rows['ranking'] == ranking].index)

                # Find nearest dspacing in the filtered df
                nearest_row = df_filtered.iloc[(df_filtered['dspacing'] - row['dspacing']).abs().argsort()[0]]

                # Assign new component, phase, and ranking
                dfp_avg.loc[index, ['components', 'phase', 'ranking']] = nearest_row[['components', 'phase', 'ranking']]
                changes_made = True  # Mark that a change was made

    return dfp_avg

def PlotGraph(imgname,pxsize,fftmodelname,f_shift,final_file,mother_folder,df,dimension_factor,binned):
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    fftname = imgname+"_FFT_final.png"
    image = Image.open(final_file+"finalfft/"+fftname).convert("L")
    arr = np.asarray(image)
    
    
     
    
    dirpath = final_file
    
    fftname = dirpath+'model/'+imgname+'_FFT_model.png'
    fftname2 = dirpath+'finalfft/'+imgname+'_FFT_final.png'
    
    #### WATERSHED SEGMENTATION
    img = cv2.imread(fftname)
    img_grey = img[:,:,0]

    sure_fg = np.uint8(img_grey)
    ret3, markers = cv2.connectedComponents(img_grey)
    #markers = markers+10

    # Now, mark the region of unknown with zero
    #markers[unknown==255] = 0
    #plt.imshow(markers, cmap='gray')   #Look at the 3 distinct regions.

    #Now we are ready for watershed filling. 
    #markers = cv2.watershed(img, markers)
    #img[markers == -1] = [0,255,255]  

    img2 = color.label2rgb(markers, bg_label=0)
    props = measure.regionprops_table(markers, intensity_image=img_grey, properties=['label','centroid','major_axis_length', 'minor_axis_length'])
    dfp = pd.DataFrame(props)
    
    #dfp=dfp[1:]
    dfp['distance'] = [np.sqrt((list(dfp['centroid-0'])[i]-511.5)*(list(dfp['centroid-0'])[i]-511.5)+(list(dfp['centroid-1'])[i]-511.5)*(list(dfp['centroid-1'])[i]-511.5)) for i in range(0,len(dfp))]
    if binned:
        try:
            inverse_dimension = (1/pxsize)/2
        except:
            print("Division by zero error")
            return
    else:
        try:
            inverse_dimension = (1/pxsize)
        except:
            print("Division by zero error")
            return
        
    pxsize_fft = inverse_dimension/1024
#     pxsize_actual = (1/pxsize)
#     pxsize_actual = (1/(4096*pxsize))*(2)
    #pxsize_actual = (1/(dimension_factor*pxsize))
    
    dfp['dspacing'] = [(x and (1/(x*pxsize_fft))*10) for x in dfp['distance']]
    print(dimension_factor)
    print(list(dfp['dspacing']))
    #display(dfp)
    #round_to_tenths = [round(num,2) for num in list(dfp['dspacing'])]
    #list_dspace = round_to_tenths[1:]
    dfp = dfp[dfp['dspacing']<5]
    dfp = dfp[dfp['dspacing']>1]
    dfp = dfp.reset_index()
    
    dfp['components'] = dfp['dspacing'].apply(getcomponent,df=df)
    dfp['phase'] = dfp['dspacing'].apply(getphase,df=df)
    dfp['ranking'] = dfp['dspacing'].apply(getranking,df=df)
    dfp['c_phase'] = dfp['components'] + '_' + dfp['phase']
    dfp['half_diagonal']= (np.sqrt(dfp['major_axis_length']*dfp['major_axis_length']+dfp['minor_axis_length']*dfp['minor_axis_length']))/2
    #df_avg = dfp.groupby(['components', 'phase', 'ranking']).agg(dspacing = ('dspacing', 'mean'), distance = ('distance', 'mean'), half_diagonal = ('half_diagonal', 'mean'))
  
    #df_avg=reassign_components(df,df_avg)
    #avg_dspace=list(df_avg['dspacing'])
    #avg_diagonal=list(df_avg['half_diagonal'])
    #avg_distance=list(df_avg['distance'])
    
#     unique_list = np.array(unique(list_dspace))
#     unique_list= [x for x in unique_list if 0 < x <= 0.3]
#     for x in unique_list:
#         temp_array = [list(dfp['dspacing'])[i] for i in (np.array(np.where(list_dspace == x))+1)[0]]
#         avg_dspace.append(np.mean(temp_array))
#         temp_array = [list(dfp['half_diagonal'])[i] for i in (np.array(np.where(list_dspace == x))+1)[0]]
#         avg_diagonal.append(np.mean(temp_array))
#         temp_array = [list(dfp['distance'])[i] for i in (np.array(np.where(list_dspace == x))+1)[0]]
#         avg_distance.append(np.mean(temp_array))
    
    #print(peaks)
    mat_df = pd.DataFrame()
    mat_dict = {'material':[],
        'value':[],
        'match_percent':[],
        'frames':[],
        'area':[],
        'intensity':[]
       }
  
    mat_df = pd.DataFrame(mat_dict)
    plt.close()
    plt.cla()
    plt.clf()
    
    count = 1
    tem_folder =final_file+'tem/' 
    background_all = Image.open(tem_folder+imgname+"_TEM.png")
    
    
    other1 = ''
    other2= ''
    temp_value2=[]
    frame = str(str(final_file.split('/')[-2]).split('_')[-1])
    unique_components = dfp['c_phase'].unique()
    for i in unique_components:
        #radiuss = i
        fftimg = cv2.imread(fftname2, cv2.IMREAD_GRAYSCALE)
        
        #value = min(list(df[1]), key=lambda x:abs(radiuss))
        #value = radiuss
        #value2 = find_nearest(np.array(df[1]),value)
        #component = str(list(dfp.loc[dfp['dspacing']==find_nearest(np.array(dfp['dspacing']),i)]['components'])[0]) + str(list(dfp.loc[dfp['dspacing']==find_nearest(np.array(dfp['dspacing']),i)]['phase'])[0])
        value = dfp[dfp['c_phase'] == str(i)]['dspacing'].mean()
        component=i
        print(component)
        #match_percent = 100 - (np.abs(value-radiuss)/value)*100
        mask = extract_component_features(markers, dfp, str(i))
        match_percent = 100
        #hh, ww = fftimg.shape[:2]
        #hh2 = hh // 2
        #ww2 = ww // 2
        # define circles
        #thickness = avg_diagonal[count-1]
        #radius = avg_distance[count-1] 
        #xc = hh2
        #yc = ww2
        #print(radiuss)
        # draw filled circle in white on black background as mask
        #mask = np.zeros_like(fftimg)
        #mask = cv2.circle(mask, (int(xc),int(yc)), int(radius), (255,255,255), thickness=int(thickness))
        mask = np.array(mask)/255
        
        mask = mask*fftimg
        
        
        path = dirpath+'allfft/'
        isExist = os.path.exists(path)

        if not isExist:
            os.makedirs(path)
        cv2.imwrite(dirpath+'allfft/fft_'+imgname+'_'+str(i)+'_.png', mask)
        
        
        
        
        mask=cv2.copyMakeBorder(mask,512,512,512,512,cv2.BORDER_CONSTANT)
        cv2.imwrite(dirpath+'mask/mask_t_'+str(count)+'_.png', mask)
        #cv2.imwrite('fft_mask.png', mask)
        im = Image.open( dirpath+'mask/mask_t_'+str(count)+'_.png' ) # .GET 258,200 [px]
        im = im.resize( ( dimension_factor,dimension_factor ) )
        #im = im.save(dirpath+'mask/mask_t_'+str(count)+'_.png')
        
        #img = cv2.imread(dirpath+'mask/mask_t_'+str(count)+'_.png',cv2.IMREAD_GRAYSCALE) # load an image
     
        im = im.save(dirpath+'mask/mask_t_'+str(count)+'_.png')
        
        img = cv2.imread(dirpath+'mask/mask_t_'+str(count)+'_.png',cv2.IMREAD_GRAYSCALE) # load an image
        
        mask = np.array(img)
        intensity_arr = np.array(img)
        fshift = f_shift*mask
        intensity_cnt = np.sum(np.array(intensity_arr))
        if(intensity_cnt>=1000):
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back).clip(0,255).astype(np.uint8)
        
            cv2.imwrite(dirpath+'mask/mask_original_'+str(i)+'.png', img_back)
            #img_back = cv2.imread(final_file+'mask/mask_temp_.png', cv2.IMREAD_GRAYSCALE)
            img_back = img_back.astype("uint8")
            
            #7x7
            
            ret, thresh = cv2.threshold(img_back,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # noise removal
            kernel = np.ones((7,7),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=0)
            
            sure_bg[sure_bg!=255]=1
            sure_bg[sure_bg==255]=0
            normalized_array = (sure_bg * 255).astype(np.uint8)

            
            area = (sum(map(sum, sure_bg)))/(sure_bg.shape[0]*sure_bg.shape[1])
        
            #sure_bg1 = sure_bg*cimg1
            #contour_img = cv2.imread("files/ground/"+str(imgname)+".png",cv2.IMREAD_GRAYSCALE)
            #contour_img = np.array(contour_img)
            #contour_img[contour_img!=255]=0
            #contour_img[contour_img==255]=1
            #sure_bg1 = sure_bg*contour_img
            cv2.imwrite(dirpath+'mask/mask_temp_.png', sure_bg)
            
            img = Image.open(dirpath+'mask/mask_temp_.png')
            img = img.convert("RGBA")
            datas = img.getdata()

            newData = []
            if(count == 1):
                color_data = (0, 0, 255, 150)
            elif(count == 2):
                color_data = (255, 0, 0, 150)
            elif(count == 3):
                color_data = (0, 255, 0, 150)
            elif(count == 4):
                color_data = (255, 0, 255, 150)
            elif(count == 5):
                color_data = (255, 255, 0, 150)
            for item in datas:
                if item[0] == 0 and item[1] == 0 and item[2] == 0:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(color_data)

            img.putdata(newData)
            path = dirpath+'mask/'
            isExist = os.path.exists(path)

            if not isExist:
                os.makedirs(path)
            mask_path = path+str(component)+'_mask.png'
            img.save(mask_path)
            img = Image.open(mask_path)
            result = Image.new('RGBA', img.size)
            
            mask_ind = Image.open(tem_folder+imgname+"_TEM.png")
            result.paste(mask_ind, (0, 0))
            #mask_ind.paste(img, (0, 0), img)
            result.paste(img, (0, 0), img)
            result.save(path+str(component)+'_img.png')
            background_all.paste(img, (0, 0), img)
        
            if value not in temp_value2:
                mat_df.loc[len(mat_df)] = [str(component), value, match_percent,frame,area,intensity_cnt]
        
            temp_value2.append(value)    
        
            count=count+1
            plt.close()
            plt.cla()
            plt.clf()
       
    
   
    #img = Image.open('/media/gany/Gany_Drive/images/'+imgname+'.png')
    
    #background_all.paste(img, (0, 0), img)
    
    mat_df.to_csv(dirpath+'matdetails/mat_details'+imgname+'.csv', index=False)
    mat_df.to_csv(mother_folder+'/heatmap_csv/'+str(final_file.split('/')[-2])+'.csv', index=False)
    background_all.save(dirpath+'mask/all_masks_'+imgname+'.png')

    
    divs = 500.0
    r = int(arr.shape[0])
    maxi=0.0
    arr1=arr
    #intensity = [0.0]*int(divs)
    #for i in range(0,r):
    #    for j in range(0,r):
    #        x=float(abs(i-int(r/2)))/float(int(r/2))
    #        y=float(abs(j-int(r/2)))/float(int(r/2))
    #        d =  float(math.sqrt(x*x + y*y))
    #        if d<=1:
    #            d = int(math.floor(d*divs))            
    #            intensity[d-1] = float(intensity[d-1]) + float(arr[i][j])
    #            if maxi<intensity[d-1]:
    #                maxi = intensity[d-1]
    intensity = radial_profile(arr, [int(r/2),int(r/2)])
    #intensity = [n / maxi for n in intensity]
    maxin = max(intensity)
    intensity = [n / maxin for n in intensity]
    intensity = np.array(intensity)
    intensity = np.subtract(intensity,min(intensity))
    intensity2 = radial_profile_avg(arr1, [int(r/2),int(r/2)])
    maxin2 = max(intensity2)
    intensity2 = [n / maxin2 for n in intensity2]
    intensity2 = np.array(intensity2)
    intensity2 = np.subtract(intensity2,min(intensity2))
    intensity3 = np.true_divide(np.add(intensity,intensity2),2)
    maxin3 = max(intensity3)
    intensity3 = [n / maxin3 for n in intensity3]
    intensity3 = np.array(intensity3)
    intensity3 = np.subtract(intensity3,min(intensity3))
    divs = len(intensity)
    radius = range(1,int(divs+1))
    fig = plt.figure(figsize = (12,10))
    axes = plt.gca()
    axes.set_ylim([0,1.1])
    xvalues = [i*100 for i in range(0,int(len(intensity)/100)+2)]
    plt.xticks(xvalues)
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)
    plt.suptitle('TEM Pattern', fontsize=20)
    plt.xlabel("Radius (a.u.)", fontsize=20)
    plt.ylabel('Intensity (a.u.)', fontsize=20)
    plt.rcParams.update({'font.size': 20})
    fig.patch.set_facecolor('xkcd:white')
    plt.plot(radius,intensity,"red",linewidth=1)
    os.makedirs(final_file+"tem_radius_graph")
    graphname = final_file+"tem_radius_graph/Radius_Graph.png"
   
    plt.savefig(graphname)
   # plt.cla()
    plt.clf()
    #plt.close('all')
    
    r2=[]
    i2=[]
    dist = 0
    theta = 0
    
    r2_modified = [((1/(x*pxsize_fft))*10) for x in radius]
    r2_modified.reverse()
    intensity=list(intensity)
    intensity.reverse()
    fig = plt.figure(figsize = (12,10))
    axes = plt.gca()
    axes.set_ylim([0,1.1])
    #xvalues = [0,1,2,3,4,5,6]
    axes.set_xlim([0,5])
    plt.xticks(np.arange(0,5))
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)
    plt.suptitle('TEM Pattern', fontsize=20)
    plt.xlabel(r"d-spacing ($\AA$)", fontsize=20)
    plt.ylabel('Intensity (a.u.)', fontsize=20)
    #for x_value in avg_dspace:
    #    plt.axvline(x_value, color='purple',linestyle='-')
    plt.rcParams.update({'font.size': 20})
    fig.patch.set_facecolor('xkcd:white')
    plt.autoscale(False)
    plt.plot(r2_modified,intensity,"red",linewidth=1)
    os.makedirs(final_file+"tem_spacing_graph")
    sgraphname = final_file+"tem_spacing_graph/Spacing_Graph.png"
    plt.savefig(sgraphname)
   # plt.cla()
    plt.clf()
   # plt.close('all')
    intensity.reverse()
    print(">>TEM Diffraction Graphs Saved")
    data = list(zip(radius,intensity))
    data = np.array(data)
    os.makedirs(final_file+"tem_rgraph_data")
    dataname=final_file+"tem_rgraph_data/rdata.csv"
    
    np.savetxt(dataname, data, delimiter=',', fmt='%r')
    intensity.reverse()
    data2 = list(zip(r2_modified,intensity))
    data2 = np.array(data2)
    os.makedirs(final_file+"tem_sgraph_data")
    dataname2=final_file+"tem_sgraph_data/sdata.csv"
    
    np.savetxt(dataname2, data2, delimiter=',', fmt='%r')
    
    sgraph_csv_name = mother_folder+'/sgraph_csv/'+str(final_file.split('/')[-2])+'.csv'
    np.savetxt(sgraph_csv_name, data2, delimiter=',', fmt='%r')
    
    print(">>TEM Diffraction Data Saved")
    
    #mask = cv2.imread('fft_mask.png')
    #mask = np.array(mask)/255
    #mask = np.resize(mask,(4096,4096,2))
    #fshift = f_shift*mask
    #f_ishift = np.fft.ifftshift(fshift)
    #simg_back = cv2.idft(f_ishift)
    #img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    
    #cv2.imwrite('fft_restored.png', img_back)
    #tempfft_im=Image.open("tem_fft/"+fftname).convert("L")
    #drawfft = ImageDraw.Draw(tempfft_im) #Draw annotation
    #drawfft.line((100,0.9*tempfft_im.size[1], 100+(tenpxlen/2),0.9*tempfft_im.size[1]), fill=255,width=20)
    #fftfontsize=int(0.04*tempfft_im.size[1])
    #font = ImageFont.truetype("font.ttf", fftfontsize)
    #drawfft.text((100,0.9*tempfft_im.size[1] - fftfontsize -30),"1/20 1/nm","white",font=font)
    #plt.imsave("tem_fft/"+fftname, tempfft_im, cmap='gray')
    print("Generating Heatmap....")
    PlotHeatmap(mother_folder)
    print("Generating Temporal map....")
    PlotTemporalmap(mother_folder)
    #Export_PDF(mother_folder)
    