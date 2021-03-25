import PIL 
import numpy as numpy
import os

from os import listdir
from matplotlib import image
from PIL import Image


# define a function that read an address of the directory 
def read_picture_data_set(adress):

	temp = [None]*(len(next(os.walk(adress))[1]))
	i = 0

	for foldername in listdir(adress):
	    loaded_images = list()
	    for filename in listdir(adress + "\\" +foldername):
	        
	        # check the image first 
	        # there are some files that are borken, we want to import them as a place holding matrix
	        size = os.stat(adress + "\\"+ foldername + "\\" + filename).st_size
	        if size != 0 : 
	            # load image
	            img_data = image.imread(adress + "\\"+ foldername + "\\" + filename)
	       
	        else: 
	            # if file is broken, just fill a place holding image
	            img_data = [  [[0,0,0] , [0,0,0]], [[0,0,0] , [0,0,0] ]]
	            
	        # store loaded image
	        loaded_images.append(numpy.array(img_data))
	        # print('> loaded %s %s' % (filename, img_data.shape))
	        
	    temp[i] = loaded_images
	    print('> loaded %s ' % (foldername))
	    i=i+1

# read all the folders of files in temp directory
# This should be the actual address of where you store the dataset in your computer
adress_temp = r"C:\Users\Administrator\Desktop\PRML\Project\fabric_data\temp"
read_picture_data_set(adress_temp)
print("Finish importing temp directory")

# read all the folders of files in trgt directory
# This should be the actual address of where you store the dataset in your computer
adress_trgt = r"C:\Users\Administrator\Desktop\PRML\Project\fabric_data\temp"
read_picture_data_set(adress_trgt)
print("Finish importing trgt directory")



import json
import glob, os

def load_fabric_data(path):
    '''
    Loads data from fabric_data folder.
    Returns:
        (list, list)
        A list of id in the file name
        A list of dictionary file containing data from json (flaw_type, bbox)

    Sample usage: fid, fdata = load_fabric_data('fabric_data/label_json/**/**.json')
    '''
    fid = []
    fdata = []
    for filename in glob.iglob(path, recursive=True):
        fid.append(filename.split('/')[-1].split('.')[0])
        with open(filename) as f:
            fdata.append(json.load(f))
    return (fid, fdata)

import numpy as numpy
import os
import cv2
from os import listdir



# a function that generate the multi-d numpy array of an image
# the image stands for the visual difference between two images 
# inputs are two image address 
def image_diff_extract(image1_address, image2_address):
    
    img2 = Image.open(image1_address)
    img1 = Image.open(image2_address)
    
    # obtain width 
    width_1, height_1 = img1.size
    width_2, height_2 = img2.size
    
    # image resizing
    resize_wdith = math.floor((width_1 + width_2)/2)
    resize_height =math.floor((height_1+ height_2)/2)
    

    Nimg1 = img1.resize((resize_wdith,resize_height))   
    Nimg2 = img2.resize((resize_wdith,resize_height))

    Nimg1 =np.array(Nimg1)
    Nimg2 =np.array(Nimg2)

    diff = cv2.absdiff(Nimg1, Nimg2)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    th = 60
    imask =  mask>th

    canvas = np.zeros_like(Nimg2, np.uint8)
     
    #make an inversion (turn it on for humans, turn it off for machines)
    # canvas = cv2.bitwise_not(canvas)

    canvas[imask] = Nimg2[imask]
    return canvas 


# Example of Reading in a image and show its difference 
first_image = r"C:\Users\Administrator\Desktop\prml_group\PRML\main\image_pair\pair_1\temp.jpg"
second_image = r"C:\Users\Administrator\Desktop\prml_group\PRML\main\image_pair\pair_1\trgt.jpg"
blob = image_diff_extract(first_image, second_image)
cv2.imwrite("result.jpg", blob)

final = Image.open(r"C:\Users\Administrator\Desktop\prml_group\PRML\main\result.jpg")
final.show()

# convert all paires of images in to image difference 
# Input: given an address where the main folder is stored
# ouput a list of multi-d numpy array (each image_diff)
# this function requires OS package 
# the input adress should be the the directory where both temp and target locate

def read_picture_to_list_image_diff(adress):
    adress_temp = adress + "\\temp"
    adress_trgt = adress + "\\trgt"
    loaded_images_rep = list()
    
   
    if not os.path.exists("my_folder_image_diff"):
        os.makedirs('my_folder_image_diff')
      
    # local saving work still in progress    
    # saving path: 
    # path= adress + "\\" + "my_folder_image_diff"
    #  print(path)
    for foldername in listdir(adress_temp):
        for filename in listdir(adress_temp +"\\" +foldername):
              
            # check the image first 
            # there are some files that are borken, we want to import them as a place holding matrix
            temp_file_adress= adress_temp + "\\"+ foldername + "\\" + filename
            trgt_file_adress= adress_trgt + "\\"+ foldername + "\\" + filename
            size_temp_file = os.stat(temp_file_adress).st_size
            size_trgt_file = os.stat(trgt_file_adress).st_size
            
            if size_temp_file != 0 and size_trgt_file != 0:
                # load both adress into the image diference function: 
                image_diff_nparray = image_diff_extract(temp_file_adress, trgt_file_adress)
                   
            else: 
                # if file is broken, just fill a place holding image
                image_diff_nparray = [  [[0,0,0] , [0,0,0]], [[0,0,0] , [0,0,0] ]]

            loaded_images_rep.append(image_diff_nparray)
            #cv2.imwrite( filename +".jpg", image_diff_nparray)
            #cv2.waitKey(0)
            
            
        print("finish converting and loading at: "+adress_temp +"\\" +foldername)
    return loaded_images_rep

# run the function here 
# put down here where you stored the fabric data folder 
adress_main = r"C:\Users\Administrator\Desktop\PRML\Project\fabric_data"
# the blob_2 will contain the list of images (images are numpy arrays of image diffeence representation)
blob_2 = read_picture_to_list_image_diff(adress_main)
