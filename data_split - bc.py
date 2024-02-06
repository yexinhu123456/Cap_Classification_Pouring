
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:53:32 2023

@author: yexin
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import re
from scipy.ndimage import gaussian_filter1d
from scipy import signal


def adjust_values(x):
    for i in range(len(x) - 100):
        if x[i] - x[i + 100] >= 1:
            x[i] = x[i + 100]
    return x

def multi_channel_lowpassfilter(data, NWn=[1, 0.04], realtime=True):
    n_samples, n_channels = data.shape
    filtered_data = np.zeros_like(data)  # Initialize an array with same shape as data
    
    for channel in range(n_channels):
        if realtime:
            single_channel_data = data[:, channel]
            filtered_channel_data = lowpassfilter(single_channel_data, NWn = NWn, realtime = realtime)
            filtered_data[:, channel] = np.squeeze(filtered_channel_data)  # Store the filtered data for this channel

            
    return filtered_data



def lowpassfilter(data, NWn=[1, 0.04], realtime=True):
    b, a = signal.butter(NWn[0], NWn[1], analog=False)
    if realtime:
        # Real time filtering
        zi = signal.lfilter_zi(b, a)
        filtered = []
        for d in data:
            s_filtered, zi = signal.lfilter(b, a, [d], zi=zi)
            filtered.append(s_filtered)
    else:
        # Forward backward filtering (no time delay, but cannot be used for real time filtering)
        filtered = np.expand_dims(signal.filtfilt(b, a, data), axis=-1)
    return np.array(filtered)



def sliding_gaussian_filter(data, window_size, sigma):
    # Pad the data to handle edges
    half_window = window_size // 2
    padded_data = np.pad(data, ((half_window, half_window), (0, 0)), mode='reflect')
    
    # Resultant array
    result = np.zeros_like(data)
    
    # Apply Gaussian filter in a sliding window
    for i in range(data.shape[0]):
        # Applying the Gaussian filter to the slice
        filtered_slice = gaussian_filter1d(padded_data[i:i + window_size], sigma=sigma, axis=0)
        
        # Assign the center of the filtered slice to the result array
        result[i] = filtered_slice[half_window]
    
    return result






weight_train = []
weight_test = []
cap_train = []
cap_test = []
num_train = []
num_test = []
weight_num_train = []
weight_num_test = []
b_train = []
b_test = []

name = ["bh_water_50", "bh_water_75", "bh_water_100", "bh_water_125",
        "bh_lentils_50", "bh_lentils_75", "bh_lentils_100", "bh_lentils_125",
        "bh_rice_50", "bh_rice_75", "bh_rice_100", "bh_rice_125",
        "bh_vinegar_50", "bh_vinegar_75", "bh_vinegar_100", "bh_vinegar_125",
        "bh_oil_50", "bh_oil_75", "bh_oil_100", "bh_oil_125"]


for i in range(20):
    k = name[i]
    os.chdir(f"C:/Users/yexin/Desktop/liquid/data/data_collection_{k}")
    # os.chdir(f"C:/Users/yexin/Desktop/liquid/data/data_collection_1012{i:02}")
    # os.chdir(f"C:/Users/yexin/Desktop/liquid/data/data_collection_09140{i}")
    
    if i == 0 or i == 1 or i == 2 or i == 3:
        num = 0
    
    elif i == 4 or i == 5 or i == 6 or i == 7:
        num = 1
        
    elif i == 8 or i == 9 or i == 10 or i == 11:
        num = 2
        
    elif i == 12 or i == 13 or i == 14 or i == 15:
        num = 3
        
    else:
        num = 4
        
        
    if i == 0 or i == 4 or i == 8 or i == 12 or i == 16:
        weight_num = 0
    
    elif i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
        weight_num = 1
        
    elif i == 2 or i == 6 or i == 10 or i == 14 or i == 18:
        weight_num = 2
        
    else:
        weight_num = 3
        



    all_pkl_files = sorted([f for f in os.listdir() if f.endswith('.pkl')])
    for idx, f in enumerate(all_pkl_files):

        
        if idx == 0: 

            data = pickle.load(open(f, "rb"))
            s = data[0][:, 8 : 18].astype(np.float32)
    

            
    
            
            weight_test.append(np.mean(data[1][-300:-1]))
            
            
            s = (s - 500) / (1000 - 500)

            cap_test.append(s)
            num_test.append(num)
            weight_num_test.append(weight_num)

            b_test.append(data[3])
            print(f"test:{f}")
          
        else:
            data = pickle.load(open(f, "rb"))
            s = data[0][:, 8 : 18].astype(np.float32)
    

            
    
            
            weight_train.append(np.mean(data[1][-300:-1]))
            
            
            s = (s - 500) / (1000 - 500)

            cap_train.append(s)
            num_train.append(num)
            weight_num_train.append(weight_num)
            b_train.append(data[3])
            print(f"train:{f}")
          
          

 

            
pickle.dump(weight_train, open(r"C:\Users\yexin\Desktop\liquid\training data\weight_train_bc.pkl", "wb"))
pickle.dump(weight_test, open(r"C:\Users\yexin\Desktop\liquid\training data\weight_test_bc.pkl", "wb"))
pickle.dump(cap_train, open(r"C:\Users\yexin\Desktop\liquid\training data\cap_train_bc.pkl", "wb"))
pickle.dump(cap_test, open(r"C:\Users\yexin\Desktop\liquid\training data\cap_test_bc.pkl", "wb"))
pickle.dump(num_train, open(r"C:\Users\yexin\Desktop\liquid\training data\num_train_bc.pkl", "wb"))
pickle.dump(num_test, open(r"C:\Users\yexin\Desktop\liquid\training data\num_test_bc.pkl", "wb"))
pickle.dump(weight_num_train, open(r"C:\Users\yexin\Desktop\liquid\training data\weight_num_train_bc.pkl", "wb"))
pickle.dump(weight_num_test, open(r"C:\Users\yexin\Desktop\liquid\training data\weight_num_test_bc.pkl", "wb"))
pickle.dump(b_train, open(r"C:\Users\yexin\Desktop\liquid\training data\b_train_bc.pkl", "wb"))
pickle.dump(b_test, open(r"C:\Users\yexin\Desktop\liquid\training data\b_test_bc.pkl", "wb"))
            

            