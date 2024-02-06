
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


name = ["water_150_final", "lentils_150_final", "rice_150_final", "vinegar_150_final", "oil_150_final"]


for i in range(5):
    k = name[i]
    os.chdir(f"C:/Users/yexin/Desktop/liquid/data/data_collection_{k}")

    num = i

    all_pkl_files = sorted([f for f in os.listdir() if f.endswith('.pkl')])
    for idx, f in enumerate(all_pkl_files):

        if idx == 0 or idx == 5:

            data = pickle.load(open(f, "rb"))
            s = data[0][:, 8 : 18].astype(np.float32)
    
            # s = multi_channel_lowpassfilter(s, NWn=[1, 0.04], realtime=True)
            
    
            
            w = data[1].astype(np.float32)
            w = lowpassfilter(w, NWn=[1, 0.02], realtime=False)
            w = np.array(w).reshape(-1).astype(np.float32)
            
            w = adjust_values(w)
            
            s = (s - 500) / (1000 - 500)
            # s = (s - np.min(s, axis=1)[:, np.newaxis]) / (np.max(s, axis=1) - np.min(s, axis=1))[:, np.newaxis]
            weight_test.append(w)
            cap_test.append(s)
            num_test.append(num)
            print(f"test:{f}")
          
        else:
            data = pickle.load(open(f, "rb"))
            s = data[0][:, 8 : 18].astype(np.float32)
            
            # s = multi_channel_lowpassfilter(s, NWn=[1, 0.04], realtime=True)
            

            
            s = (s - 500) / (1000 - 500)
            
            # s = (s - np.min(s, axis=1)[:, np.newaxis]) / (np.max(s, axis=1) - np.min(s, axis=1))[:, np.newaxis]
            w = data[1].astype(np.float32)
            w = lowpassfilter(w, NWn=[1, 0.02], realtime=False)
            w = np.array(w).reshape(-1).astype(np.float32)
            w = adjust_values(w)
            
            weight_train.append(w)
            cap_train.append(s)
            num_train.append(num)
            print(f"train:{f}")
          
          

 

            
pickle.dump(weight_train, open(r"C:\Users\yexin\Desktop\liquid\training data\weight_train.pkl", "wb"))
pickle.dump(weight_test, open(r"C:\Users\yexin\Desktop\liquid\training data\weight_test.pkl", "wb"))
pickle.dump(cap_train, open(r"C:\Users\yexin\Desktop\liquid\training data\cap_train.pkl", "wb"))
pickle.dump(cap_test, open(r"C:\Users\yexin\Desktop\liquid\training data\cap_test.pkl", "wb"))
pickle.dump(num_train, open(r"C:\Users\yexin\Desktop\liquid\training data\num_train.pkl", "wb"))
pickle.dump(num_test, open(r"C:\Users\yexin\Desktop\liquid\training data\num_test.pkl", "wb"))        
            

            