# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:57:06 2023

@author: yexin
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import torch.nn.functional as F
import random



class data_train_s(Dataset):
    def __init__(self, cap):
        self.cap = cap
        # Assuming each item in cap can give at least 20 readings
        self.total_sequences = sum([len(item) - 19 for item in cap])

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        cumulative_lengths = np.cumsum([len(item) - 19 for item in self.cap])
        num = np.searchsorted(cumulative_lengths, idx + 1)

        # If it's the first sequence, start from 0, else determine the starting point based on idx
        if num == 0:
            start_idx = idx
        else:
            start_idx = idx - cumulative_lengths[num - 1]

        capacitance = self.cap[num][start_idx: start_idx + 20]

        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(start_idx)

    
    
class data_test_s(Dataset):
    def __init__(self, cap):
        self.cap = cap
        # Assuming each item in cap can give at least 20 readings
        self.total_sequences = sum([len(item) - 19 for item in cap])

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        cumulative_lengths = np.cumsum([len(item) - 19 for item in self.cap])
        num = np.searchsorted(cumulative_lengths, idx + 1)

        # If it's the first sequence, start from 0, else determine the starting point based on idx
        if num == 0:
            start_idx = idx
        else:
            start_idx = idx - cumulative_lengths[num - 1]

        capacitance = self.cap[num][start_idx: start_idx + 20]

        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(start_idx)





class data_train_10(Dataset):
    def __init__(self, cap):
        self.cap = cap
        # Assuming each item in cap can give at least 20 readings
        self.total_sequences = sum([len(item) - 9 for item in cap])

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        cumulative_lengths = np.cumsum([len(item) - 9 for item in self.cap])
        num = np.searchsorted(cumulative_lengths, idx + 1)

        # If it's the first sequence, start from 0, else determine the starting point based on idx
        if num == 0:
            start_idx = idx
        else:
            start_idx = idx - cumulative_lengths[num - 1]

        capacitance = self.cap[num][start_idx: start_idx + 10]

        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(start_idx)
    
class data_test_10(Dataset):
    def __init__(self, cap):
        self.cap = cap
        # Assuming each item in cap can give at least 20 readings
        self.total_sequences = sum([len(item) - 9 for item in cap])

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        cumulative_lengths = np.cumsum([len(item) - 9 for item in self.cap])
        num = np.searchsorted(cumulative_lengths, idx + 1)

        # If it's the first sequence, start from 0, else determine the starting point based on idx
        if num == 0:
            start_idx = idx
        else:
            start_idx = idx - cumulative_lengths[num - 1]

        capacitance = self.cap[num][start_idx: start_idx + 10]

        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(start_idx)

    













class data_train(Dataset):
    def __init__(self, cap):

        
        self.cap = cap


    def __len__(self):
        return int(100000)

    
    def __getitem__(self, idx):
        num = np.random.randint(len(self.cap))
        cap_list = self.cap[num]
        start_idx = random.randint(0, cap_list.shape[0] - 20)
        capacitance = cap_list[start_idx: start_idx + 20] 

        
        
        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(start_idx)

    
    
class data_test(Dataset):
    def __init__(self, cap):

        
        self.cap = cap


    def __len__(self):
        return int(10000)

    
    def __getitem__(self, idx):
        num = np.random.randint(len(self.cap))
        cap_list = self.cap[num]
        start_idx = random.randint(0, cap_list.shape[0] - 20)
        capacitance = cap_list[start_idx: start_idx + 20]
        

        
        
        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(start_idx) 
    
    

class data_train_simple(Dataset):
    def __init__(self, cap):

        
        self.cap = cap


    def __len__(self):
        return int(100000)

    
    def __getitem__(self, idx):
        num = np.random.randint(len(self.cap))
        cap_list = self.cap[num]
        start_idx = random.randint(0, cap_list.shape[0] - 20)
        capacitance = cap_list[start_idx: start_idx + 20] 

        
        
        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(start_idx)
    

    
    
class data_test_simple(Dataset):
    def __init__(self, cap):

        
        self.cap = cap


    def __len__(self):
        return int(10000)

    
    def __getitem__(self, idx):
        num = np.random.randint(len(self.cap))
        cap_list = self.cap[num]
        start_idx = random.randint(0, cap_list.shape[0] - 20)
        capacitance = cap_list[start_idx: start_idx + 20]
        

        
        
        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(start_idx)
    
    
    
    
class data_train_d(Dataset):
    def __init__(self, seq, label):
        
        
        self.seq = seq
        self.label = label
    
    
    def __len__(self):
        
        return len(self.seq)
        
        
        
    def __getitem__(self, idx):

        
        
        return torch.tensor(self.seq[idx]), torch.tensor(self.label[idx])
    
    
class data_test_d(Dataset):
    def __init__(self, seq, label):
        
        
        self.seq = seq
        self.label = label
    
    
    def __len__(self):
        
        return len(self.seq)
        
        
        
    def __getitem__(self, idx):

        
        
        return torch.tensor(self.seq[idx]), torch.tensor(self.label[idx])
    










class data_train_bc(Dataset):
    def __init__(self, cap, b):
        self.cap = cap
        self.b = b
        # Assuming each item in cap can give at least 20 readings
        self.total_sequences = sum([len(item) - 9 for item in cap])

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        cumulative_lengths = np.cumsum([len(item) - 9 for item in self.cap])
        num = np.searchsorted(cumulative_lengths, idx + 1)

        # If it's the first sequence, start from 0, else determine the starting point based on idx
        if num == 0:
            start_idx = idx
        else:
            start_idx = idx - cumulative_lengths[num - 1]

        capacitance = self.cap[num][start_idx: start_idx + 10]
        bh = self.b[num][start_idx: start_idx + 10]
        if np.mean(bh) > 0.01:
            a = 1
        else:
            a = 0

        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(a)
    
    
    
class data_test_bc(Dataset):
    def __init__(self, cap, b):
        self.cap = cap
        self.b = b
        # Assuming each item in cap can give at least 20 readings
        self.total_sequences = sum([len(item) - 9 for item in cap])

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        cumulative_lengths = np.cumsum([len(item) - 9 for item in self.cap])
        num = np.searchsorted(cumulative_lengths, idx + 1)

        # If it's the first sequence, start from 0, else determine the starting point based on idx
        if num == 0:
            start_idx = idx
        else:
            start_idx = idx - cumulative_lengths[num - 1]

        capacitance = self.cap[num][start_idx: start_idx + 10]
        bh = self.b[num][start_idx: start_idx + 10]
        if np.mean(bh) > 0.01:
            a = 1
        else:
            a = 0

        return torch.tensor(capacitance), torch.tensor(num), torch.tensor(a)






