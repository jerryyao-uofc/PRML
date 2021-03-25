import json
import glob, os

import numpy as np
import cv2

from PIL import Image
#numerical
import math

from matplotlib import image
from matplotlib import pyplot

import PIL 

import os
from os import listdir

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
        #print("filename in load_fabric_data is ")
       # print(filename)
        filename = filename.replace('\\' ,'/')
        
        fid.append(filename.split('/')[-1].split('.')[0])
        with open(filename) as f:
            fdata.append(json.load(f))
    return (fid, fdata)

def extract_label_grouping(fdata):
    '''
    Generates lists of labels according to different groupings.
    Type 1 grouping: original
    Type 2 grouping: 6-12 as group 6, 13 as group 7, 14 as group 8
    Type 3 grouping: only take 1,2,5 and 13
    '''
    ftype1 = [] #original
    ftype2 = [] #6-12 as group 6, 13 as group 7, 14 as group 8
    ftype2_dict = {num:6 for num in range(6, 13)}
    ftype2_dict.update({num:num for num in range(6)})
    ftype2_dict[13] = 7
    ftype2_dict[14] = 8 
    for i in fdata:
        ftype1.append(i['flaw_type'])
        ftype2.append(ftype2_dict[i['flaw_type']])
    return ftype1, ftype2
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


def load_fabric_images(path, fids, fdata, ftype):
    path += '**/**.jpg'
    labels = []
    imgs = []

    print(path)
    #random.sample(list(glob.iglob(path, recursive=True)), 50)
    for filename in glob.iglob(path, recursive=True):
        #find info about the image
        filename = filename.replace('\\' ,'/')


        fid = filename.split('/')[-1].split('.')[0]
     
        info = fdata[fids.index(fid)]
        
        filename_trgt = filename.replace("temp", "trgt")
        #filename_trgt = r"C:/Users/Administrator/Desktop/PRML/Project/fabric_data/trgt" + "/" +fid +'.jpg'
        #get image
        size1 = os.stat(filename).st_size
        size2 = os.stat(filename_trgt).st_size
        if (size1 != 0) and (size2 != 0): 
            #load image
            img_data_temp = image.imread(filename)
            img_data_temp_area = img_data_temp[info['bbox']['y0']:info['bbox']['y1'], info['bbox']['x0']:info['bbox']['x1']]
            img_data_tgrt = image.imread(filename_trgt)
            img_data_tgrt_area = img_data_tgrt[info['bbox']['y0']:info['bbox']['y1'], info['bbox']['x0']:info['bbox']['x1']]
            
            # extract the difference between two images 
            diff= image_diff_extract(filename ,filename_trgt)
            #diff = diff[info['bbox']['y0']:info['bbox']['y1'], info['bbox']['x0']:info['bbox']['x1']]
            
           
            # print(filename)
            if (diff.shape[0] !=0) and (diff.shape[1] !=0):
                if (img_data_temp_area.shape[1] !=0) and (img_data_temp_area.shape[0] !=0):
                    if (img_data_tgrt_area.shape[1] !=0) and (img_data_tgrt_area.shape[0] !=0):
                        # print("the size is ")
                        # print(diff.shape)
                        height = diff.shape[0]
                        width = diff.shape[1] 
                        img_data_temp = cv2.resize(img_data_temp , (width, height)) 
                        # print("the size 1 is ")
                        # print(img_data_temp.shape)
                        img_data_tgrt = cv2.resize(img_data_tgrt, (width, height)) 
                        # print("the size 2 is ")
                        # print(img_data_tgrt.shape)
                        img_data_temp_area =cv2.resize(img_data_temp_area, (width, height))
                        # print("the size 3 is ")
                        # print(img_data_temp_area.shape)
                        img_data_tgrt_area =cv2.resize(img_data_tgrt_area, (width, height))
                        # print("the size 4 is ")
                        # print( img_data_tgrt_area.shape)

                        imgs.append(np.concatenate([img_data_temp, img_data_tgrt, diff, img_data_temp_area, img_data_tgrt_area], axis = 2))
                        labels.append(ftype[fids.index(fid)])

       
            # if img_data_temp.shape == img_data_tgrt.shape:
            #     #append image
            #     imgs.append(np.concatenate([img_data_temp, img_data_tgrt], axis = 2))
            #     #append label
            #     labels.append(ftype[fids.index(fid)])
    return (labels, imgs)


# path = r"C:/Users/Administrator/Desktop/PRML/Project/fabric_data/label_json/**/**.json"

# fids, fdata = load_fabric_data(path)
# ftype1, ftype2 = extract_label_grouping(fdata)

# path = r"C:/Users/Administrator/Desktop/PRML/Project/fabric_data/temp/"
# labels, imgs = load_fabric_images(path, fids, fdata, ftype2)