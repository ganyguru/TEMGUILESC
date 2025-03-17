from __main__ import *
from skimage.transform import resize
import tensorflow as tf
import Graph
from Graph import PlotGraph
from tensorflow.keras.models import load_model
from scipy import ndimage as nd

def create_fft(dm3filename,factor,final_file,df,model):
    img_dim = (1024,1024)
    dm3f = dm3.DM3(dm3filename)
    dm3file = dm3filename.split('\\')[-1].split(".")[0]
    image = dm3f.imagedata
    dimension_factor = image.shape[1]
    tem_folder =final_file+'tem' 
    os.makedirs(tem_folder)
    halfft_folder =final_file+'half_fft'
    os.makedirs(halfft_folder)
    model_folder =final_file+'model'
    os.makedirs(model_folder)
    finalfft_folder =final_file+'finalfft'
    os.makedirs(finalfft_folder)
    os.makedirs(final_file+'mask')
    os.makedirs(final_file+'matdetails')
    temname = tem_folder+"/"+dm3file+"_TEM.png"
    plt.imsave(temname, image, cmap='gray')
    IMG_CHANNELS = 3
    imgsize_h = 1024
    imgsize_w = 512
    imgsize=512
    X_test = np.zeros((1, imgsize, imgsize, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    img = cv2.imread(temname)
    dim = (imgsize, imgsize)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #gaussian_img = nd.gaussian_filter(resized, sigma=3)
    # resize image
    stemname = tem_folder+"/"+dm3file+"_sTEM.png"
    plt.imsave(stemname,resized,cmap='gray')
 
 
    
    img = cv2.imread(temname) # load an image
    
    size_str = dm3f.pxsize
    pxsize = float(size_str[0])*0.1
    print(pxsize)
   
    print(">>TEM image Saved")
    img = img[:,:,2] # blue channel
    r = img.shape[0]
    f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    f_shift = np.fft.fftshift(f)
    f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
    f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
    f_bounded = 20 * np.log(f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    f_img=f_img[int(r/4):int((3*r)/4),int(r/4):int((3*r)/4)]
    mini = 255
    r=int(r/2)
    a= np.array(f_img[0:int((8*r)/3),0:int((8*r)/3)])
    s = a.mean()
    #for i in range(0,r):
    #    for j in range(0,r):
    #        if f_img[i][j]-s<=0:
    #            f_img[i][j]=0
    #        else:
    #            f_img[i][j]=f_img[i][j]-s
    #        mindex = min(i,j)
            #f_img[i][j] = f_img[i][j] * (1 - (math.exp(abs(mindex-1024)-1024))/math.exp(1))
    #        f_img[i][j] = int(f_img[i][j] * (1 - (math.exp(abs(mindex-int(r/2))-int(r/2)))/math.exp(1)))
    f_img=np.subtract(f_img,s)
    f_img = f_img.clip(min=0)

    f_img = f_img.astype(int)
    maxi = f_img.max()
    maxmult = 255/maxi
    f_img = f_img*maxmult
    
    fft_folder =final_file+'fft' 
    os.makedirs(fft_folder)
    fftsavename = fft_folder+"/"+dm3file+"_FFT.png"
    graphinput = dm3file
    cv2.imwrite(fftsavename,f_img)
    tweak_image = cv2.imread(fftsavename)
    alpha = 3 # Contrast control (1.0-3.0)
    beta = 11 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(tweak_image, alpha=alpha, beta=beta)

    cv2.imwrite(fftsavename,adjusted)
    #pre_pro_fft = cv2.imread(temp_save)
    img_fft_temp = cv2.imread(fftsavename)
    img_fft_temp=cv2.copyMakeBorder(img_fft_temp, int((2048-img_fft_temp.shape[0])/2),int((2048-img_fft_temp.shape[0])/2),int((2048-img_fft_temp.shape[0])/2),int((2048-img_fft_temp.shape[0])/2), cv2.BORDER_CONSTANT,value=(0,0,0))
    cv2.imwrite(fftsavename,img_fft_temp)
    #img_x = cv2.imread(temp_save)[:,:,:IMG_CHANNELS]
    sizes_test = []
    IMG_WIDTH = 512
    IMG_HEIGHT = 1024
    IMG_CHANNELS = 3
    X_test = np.zeros((1,IMG_HEIGHT,IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    #print(img_x.shape)
    #img = np.expand_dims(img,axis=2)
 
    
    image = Image.open(fftsavename).convert("L")
    fft_sizef = np.array(image).shape[0]
    half_img = image.crop((0,0,image.size[0]/2,image.size[1]))
    arr_half_img = np.asarray(half_img)
    temp_save_h = halfft_folder +"/"+dm3file+"_FFT_h.png"
    plt.imsave(temp_save_h,arr_half_img,cmap = 'gray')
    
    img = cv2.imread(temp_save_h,0)
    
    sizes_test.append([img.shape[0], img.shape[1]])
    #img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    img = nd.gaussian_filter(img, sigma=3)
    plt.imsave(temp_save_h,img,cmap = 'gray')
    img = cv2.imread(temp_save_h,0)
    img = resize(img, (IMG_HEIGHT,IMG_WIDTH), mode='constant', preserve_range=True)
    
    plt.imsave(temp_save_h,img,cmap = 'gray')
    
    img_x = cv2.imread(temp_save_h)[:,:,:IMG_CHANNELS]
    
    
    
    
    X_test[0] = img_x
    preds_test = model.predict(X_test, verbose=1)
    
    fftmodelname = model_folder +"/"+dm3file+"_FFT_model.png"
    fftfinalname = finalfft_folder +"/"+dm3file+"_FFT_final.png"
    preds_test= (preds_test > 0.5).astype(np.uint8)
    plt.imsave(fftmodelname,(np.squeeze(preds_test[0])),cmap='gray')
    fft_half_img = cv2.imread(fftmodelname,0)
    flippedimage= cv2.flip(fft_half_img, 1)
    flippedimage= cv2.flip(flippedimage, 0)
    final_image = np.concatenate((fft_half_img, flippedimage), axis=1)
    plt.imsave(fftmodelname,final_image,cmap='gray')
    
    fname = fftsavename
    fname2 = fftmodelname
    
    image = Image.open(fname).convert("L")
    image2 = Image.open(fname2).convert("L")
    dim = (1024, 1024)
    im1 = image.resize(dim)
    arr = np.asarray(im1)
    arr2 = np.asarray(image2)
    arr2 = arr2/255
    final_array = arr*np.array(arr2)
    plt.imsave(fftfinalname, final_array, cmap='gray')
    
    
    print(">>FFT image Generated")
    del model
    plt.clf()
    #plt.close('all')

    PlotGraph(graphinput,pxsize,fftmodelname,dft_shift,final_file,df,dimension_factor)
    
