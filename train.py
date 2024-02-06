# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:42:04 2023

@author: yexin
"""
import math
import torch.nn as nn
import time
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
from model import Transformer, MLP, CNN1D, MLP_2, MLP_2_resnet, MLP_2_10, MLP_2_10_resnet, MLP_2_10_3, MLP_2_10_33, MLP_2_10_3_tanh, MLP_2_10_resnet_class
from dataloader import data_train, data_test, data_train_s, data_test_s, data_train_10, data_test_10
import torch
from progressbar import ProgressBar
from tqdm import tqdm
# import streamlit as st
import torch.nn.functional as F
import pickle
import random
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import itertools


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_dir', type=str, default='C:/Users/yexin/Desktop/liquid/', help='Please change to your experiment path')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--num_warmup_steps', type=int, default=100, help='num_warmup_steps')
parser.add_argument('--epoch', type=int, default=50, help='The time steps you want to subsample the dataset to,100')
parser.add_argument('--train_continue', type=bool, default= False, help='Set true if continue to train')
parser.add_argument('--test', type = bool, default = True, help = "running on test set")
parser.add_argument('--train', type = bool, default = False, help = "running on train set")
args = parser.parse_args()



if not os.path.exists(args.exp_dir + 'ckpts'):
    os.makedirs(args.exp_dir + 'ckpts')
    
    
    
use_gpu = True
device = 'cuda'


def data_process(data):
    slices = []
    
    # Loop over each sample
    for i in tqdm(range(len(data))):
        # Loop over each index in the range [500, 1500]
        sample = data[i]
        sample = torch.tensor(sample)
        for idx in range(500, len(sample) - 500):
            slices_i = []
            # Calculate the start index based on the ending index
            start = idx % 10 + 1
            while start + 10 <= idx + 1:
                slice_data = sample[start:start + 10]
                slices_i.append(slice_data)
                start += 10
            
            slices_i = [torch.stack(slices_i)] + [idx] + [i]
            slices.append(slices_i)
        
    return slices



class InverseTimeDecayScheduler:
    def __init__(self, initial_lr, decay_rate, decay_step):
        """
        :param initial_lr: Initial learning rate
        :param decay_rate: Decay rate
        :param decay_step: Decay step
        """
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.step_count = 0
    
    def get_lr(self):
        """
        Get the current learning rate
        """
        lr = self.initial_lr / (1 + self.decay_rate * (self.step_count / self.decay_step))
        return lr
    
    def step(self):
        """
        Update the step count
        """
        self.step_count += 1

class CosineAnnealingWarmRestarts:
    def __init__(self, initial_lr, T_0, T_mult=1, eta_min=0):
        """
        :param initial_lr: Initial learning rate
        :param T_0: Number of iterations for the first cosine annealing cycle
        :param T_mult: Factor by which to multiply T_0 after each cycle
        :param eta_min: Minimum learning rate
        """
        self.initial_lr = initial_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.current_iteration = 0
        self.current_T = T_0
    
    def step(self):
        """
        Update the learning rate and return the new value
        """
        lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.current_iteration / self.current_T))
        
        self.current_iteration += 1
        if self.current_iteration >= self.current_T:
            self.current_iteration = 0
            self.current_T *= self.T_mult
        
        return lr


def get_class(label, nums):
    classes = []
    for i in range(len(nums)):
        classes.append(torch.tensor(label[nums[i]]))
        

    return torch.stack(classes).to(device)



def linear_interpolation(full_time, time, weight):
    interpolated_weights = np.interp(full_time, time, weight)
    return interpolated_weights

def init_weights_kaiming(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def accumulated_sums(arr):
    result = [arr[0]] if arr.size else []
    for i in range(1, len(arr)):
        result.append(result[i - 1] + arr[i])
    return np.array(result)

def random_slices(data):
    # Ensure data has the second dimension of size 10
    assert data.shape[1] == 10, "Data should have shape (_, 10)"

    # Randomly select a starting index from [0, 20]
    start_idx = np.random.randint(0, 21)

    # Initialize a list to store slices
    slices = []
    final = random.randint(500, len(data) - 500)
    # Slice data in increments of 20 from the starting index
    while start_idx + 10 <= final:
        slices.append(data[start_idx:start_idx+10])
        final_idx = start_idx+9
        start_idx += 10
        
    # If there are some leftovers at the end, add them to the slices


    return np.array(slices), final_idx

def slices(data):
    # Ensure data has the second dimension of size 10
    assert data.shape[1] == 10, "Data should have shape (_, 10)"

    # Randomly select a starting index from [0, 20]
    start_idx = np.random.randint(0, 21)

    # Initialize a list to store slices
    slices = []
    # Slice data in increments of 20 from the starting index
    while start_idx + 10 <= len(data):
        slices.append(data[start_idx:start_idx+10])
        start_idx += 10
        
    # If there are some leftovers at the end, add them to the slices


    return np.array(slices)


def first_slices(data):
    # Ensure data has the second dimension of size 10
    assert data.shape[1] == 10, "Data should have shape (_, 10)"

    # Randomly select a starting index from [0, 20]
    start_idx = 0

    # Initialize a list to store slices
    slices = []
    final_idx = []
    # Slice data in increments of 20 from the starting index
    while start_idx + 10 <= len(data):
        slices.append(data[start_idx:start_idx+10])
        final_idx.append(start_idx + 9)
        start_idx += 10
        
    # If there are some leftovers at the end, add them to the slices


    return np.array(slices), np.array(final_idx)


    

def interpolate_weights(W_list, offsets, indices, nums):
    # Ensure that offsets, indices, and nums have the same batch size
    batch_size = offsets.shape[0]
    assert batch_size == indices.shape[0] == nums.shape[0], "Batch sizes should be equal"

    results = []

    for i in range(batch_size):
        W = W_list[nums[i]]
        
        # Get integer and decimal parts
        int_offset = int(offsets[i] + indices[i])
        delta = offsets[i] + indices[i] - int_offset

        # Ensure we're not at the boundary of the tensor
        if int_offset >= len(W) - 1:
            results.append(torch.tensor(W[-1]).to(device))
        else:
            W_interp = (1 - delta) * W[int_offset] + delta * W[int_offset + 1]
            results.append(W_interp)

    return torch.stack(results)

def interpolate_weight(W_list, offset, idx, num):
    W = W_list[num]
    int_offset = int(offset + idx)
    delta = offset + idx - int_offset
    if int_offset >= len(W) - 1:
        result = torch.tensor(W[-1]).to(device)
    else:
        result = (1 - delta) * W[int_offset] + delta * W[int_offset + 1]
        
    return result


if args.train:

    train_cap = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\cap_train.pkl", "rb"))
    train_dataset = data_train_10(train_cap)
    train_weight = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\weight_train.pkl", "rb"))
    train_class = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\num_train.pkl", "rb"))
    
    print("finish load training data")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0, pin_memory=True)
    

    val_cap = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\cap_test.pkl", "rb"))
    val_weight = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\weight_test.pkl", "rb"))
    val_class = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\num_test.pkl", "rb"))
    
    val_dataset = data_test_10(val_cap)
    print("finish load val data")
    
    
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False, num_workers=0)

if args.test:
    test_cap = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\cap_test.pkl", "rb"))
    test_weight = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\weight_test.pkl", "rb"))
    test_class = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\num_test.pkl", "rb"))
    
    test_dataset = data_test_10(test_cap)
    print("finish load test data")
    
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False, num_workers=0)

if __name__ == '__main__':
    if args.train:
        np.random.seed(0)
        torch.manual_seed(412)
    
        model = MLP_2_10_resnet_class()
        # model.apply(init_weights_kaiming)
        model.to(device)
        if args.train_continue:
            checkpoint = torch.load(r"C:\Users\yexin\Desktop\liquid\ckpts\150_0.path.tar")
            model.load_state_dict(checkpoint['model_state_dict'])
    
        
    
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay, betas=(0.9, 0.999))
        # T_0 = args.num_warmup_steps  # You can adjust this to your needs
        # scheduler = CosineAnnealingWarmRestarts(initial_lr=args.lr, T_0=T_0)

        # scheduler = InverseTimeDecayScheduler(initial_lr=args.lr, decay_rate=1, decay_step=1500)
        scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, len(train_dataloader) * args.epoch)
        criterion = nn.MSELoss()
        relu = nn.ReLU()
    
    
        best_val_1 = np.inf
        best_val_3 = np.inf
        for epoch in tqdm(range(args.epoch)):
    
            loss_delta_weight_list = []
            loss_sum_weight_list = []
            loss_random_weight_list = []
            loss_p = []
            loss_o = []
    
            
            
            print ('here')
            
            bar = ProgressBar(max_value=len(train_dataloader))
    
            for i_batch, sample_batched in bar(enumerate(train_dataloader, 0)):
                model.train(not args.test)
    
    
    
                
                cap = sample_batched[0].to(device, non_blocking = True)
                num = sample_batched[1].to(device, non_blocking = True)
                start_index = sample_batched[2].to(device, non_blocking = True)
                
                
    
    
                
    
                with torch.set_grad_enabled(not args.test):
                    label_1 = get_class(train_class, num)
    
                    weight_1, offset_1, offset_2 = model(cap, label_1)
                    
                        
                    
                    
                    weight_delta = interpolate_weights(train_weight, offset_2, start_index + 10, num) - interpolate_weights(train_weight, offset_1, start_index, num)
                    weight_delta[weight_delta < 0.3] = 0
                    loss_1 = criterion(weight_1, weight_delta)
    
                    
                    
                    loss_2 = torch.tensor(0, dtype = torch.float).to(device)
                    # loss_3 = torch.tensor(0, dtype = torch.float).to(device)
                    loss_5 = torch.tensor(0, dtype = torch.float).to(device)
                    for i in range(len(num)):
                        cap_sliced, final_idx = random_slices(train_cap[num[i]])
                        label_2 = torch.full((len(cap_sliced),), train_class[num[i]]).to(device)
                        weight_2, offset_11, offset_22 = model(torch.tensor(cap_sliced).to(device, non_blocking = True), label_2)
                        # cap_sliced_3 = slices(train_cap[num[i]])
                        # weight_3, offset_111, offset_222 = model(torch.tensor(cap_sliced_3).to(device, non_blocking = True))
                        final_weight = interpolate_weight(train_weight, offset_22[-1], final_idx, num[i])
                        loss_2 += criterion(torch.sum(weight_2), torch.tensor(final_weight).to(device))
                        # loss_3 += criterion(torch.sum(weight_3), torch.tensor(train_weight[num[i]][-1]).to(device))
                        loss_5 += criterion(offset_22[0: -1], offset_11[1:])
                    
                    loss_2 = loss_2 / len(num)
                    # loss_3 = loss_3 / len(num)
                    loss_4 = torch.relu(15 - offset_1).mean() + torch.relu(15 - offset_2).mean()
                    loss_5 = loss_5 / len(num)
                    
                    loss = loss_1 + 0.2 * loss_2 + loss_4 + 0.1 * loss_5
                    if not args.test:
    
                        optimizer.zero_grad()
                        loss.backward()
                        # learning_rate = scheduler.get_lr()
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = learning_rate
                        optimizer.step()
                        scheduler.step()
                        
                        
                        
                loss_delta_weight_list.append(loss_1.data.item())
                # loss_sum_weight_list.append(loss_3.data.item())
                loss_random_weight_list.append(loss_2.data.item())
                loss_p.append(loss_4.data.item()) 
                loss_o.append(loss_5.data.item())
                
                if i_batch % 50 ==0 and i_batch > 0:
                    print(f"loss_1: {np.array(np.mean(loss_delta_weight_list))}")
                    print(f"loss_2: {np.array(np.mean(loss_random_weight_list))}")
                    # print(f"loss_3: {np.array(np.mean(loss_sum_weight_list))}")
                    print(f"loss_4: {np.array(np.mean(loss_p))}")
                    print(f"loss_5: {np.array(np.mean(loss_o))}")
                    
                    
                    
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': np.array(loss.data.item()),},
                     args.exp_dir + 'ckpts/' + str(epoch) + '.path.tar')
                    
                    
                
                
                
                
            val_delta_weight_list = []
            val_sum_weight_list = []   
            bar = ProgressBar(max_value=len(val_dataloader))     
            for i_batch, sample_batched in bar(enumerate(val_dataloader, 0)):
                model.eval()
    
    
    
                
                cap = sample_batched[0].to(device, non_blocking = True)
                num = sample_batched[1].to(device, non_blocking = True)
                start_index = sample_batched[2].to(device, non_blocking = True)
                
                
    
    
                
    
                with torch.set_grad_enabled(False):
                    label_1 = get_class(val_class, num)
                    weight_1, offset_1, offset_2 = model(cap, label_1)
                    
                        
                    
                    
                    weight_delta = interpolate_weights(val_weight, offset_2, start_index + 10, num) - interpolate_weights(val_weight, offset_1, start_index, num)
                    weight_delta[weight_delta < 0.3] = 0
                    loss_1 = criterion(weight_1, weight_delta)
                    
                    val_delta_weight_list.append(loss_1.data.item())
                    
            loss_3 = torch.tensor(0, dtype = torch.float).to(device)
            for i in range(len(val_cap)):
                cap_sliced_3 = slices(val_cap[i])
                label_3 = torch.full((len(cap_sliced_3),), val_class[i]).to(device)
                weight_3, offset_111, offset_222 = model(torch.tensor(cap_sliced_3).to(device, non_blocking = True), label_3)
                loss_3 += criterion(torch.sum(weight_3), torch.tensor(val_weight[i][-1]).to(device))
            
            loss_3 = loss_3 / len(val_cap)

                
                    
                    
            
            val_sum_weight_list.append(loss_3.data.item())

            print(f"loss_1: {np.array(np.mean(val_delta_weight_list))}")
            print(f"loss_3: {np.array(np.mean(val_sum_weight_list))}")   
            
            if np.array(np.mean(val_delta_weight_list)) < best_val_1:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),},
                 args.exp_dir + 'ckpts/' + 'best_1_class' + '.path.tar')
                    
                best_val_1 = np.array(np.mean(val_delta_weight_list))
                    
            if np.array(np.mean(val_sum_weight_list)) < best_val_3:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),},
                 args.exp_dir + 'ckpts/' + 'best_3_class' + '.path.tar')
                    
                best_val_3 = np.array(np.mean(val_sum_weight_list))

    
    elif args.test:
        np.random.seed(0)
        torch.manual_seed(99)
    
        model = MLP_2_10_resnet_class()
        model.to(device)
        checkpoint = torch.load(r"C:\Users\yexin\Desktop\liquid\ckpts\150_2.path.tar")
        model.load_state_dict(checkpoint['model_state_dict'])
    

        criterion = nn.L1Loss()
        relu = nn.ReLU()
         


        loss_delta_weight_list = []
        loss_sum_weight_list = []
        loss_random_weight_list = []
        loss_p = []
        loss_o = []

        
        
        print ('here')
        
        bar = ProgressBar(max_value=len(test_dataloader))

        for i_batch, sample_batched in bar(enumerate(test_dataloader, 0)):
            model.eval()



            
            cap = sample_batched[0].to(device, non_blocking = True)
            num = sample_batched[1].to(device, non_blocking = True)
            start_index = sample_batched[2].to(device, non_blocking = True)
            
            


            

            with torch.set_grad_enabled(False):
                label_1 = get_class(test_class, num)
                weight_1, offset_1, offset_2 = model(cap, label_1)
                
                    
                
                
                weight_delta = interpolate_weights(test_weight, offset_2, start_index + 10, num) - interpolate_weights(test_weight, offset_1, start_index, num)
                weight_delta[weight_delta < 0.3] = 0
                loss_1 = criterion(weight_1, weight_delta)
    
                
                
                
                

                    
                    
            loss_delta_weight_list.append(loss_1.data.item())
            
            
        loss_2 = torch.tensor(0, dtype = torch.float).to(device)
        loss_3 = torch.tensor(0, dtype = torch.float).to(device)
        loss_5 = torch.tensor(0, dtype = torch.float).to(device)
        for i in range(len(test_cap)):
            cap_sliced, final_idx = random_slices(test_cap[i])
            label_2 = torch.full((len(cap_sliced),), test_class[i]).to(device)
            weight_2, offset_11, offset_22 = model(torch.tensor(cap_sliced).to(device, non_blocking = True), label_2)
            cap_sliced_3 = slices(test_cap[i])
            label_3 = torch.full((len(cap_sliced_3),), test_class[i]).to(device)
            weight_3, offset_111, offset_222 = model(torch.tensor(cap_sliced_3).to(device, non_blocking = True), label_3)
            final_weight = interpolate_weight(test_weight, offset_22[-1], final_idx, i)
            loss_2 += criterion(torch.sum(weight_2), torch.tensor(final_weight).to(device))
            loss_3 += criterion(torch.sum(weight_3), torch.tensor(test_weight[i][-1]).to(device))
            loss_5 += criterion(offset_222[0: -1], offset_111[1:])
        
        print(offset_11)
        loss_2 = loss_2 / len(test_cap)
        loss_3 = loss_3 / len(test_cap)
        loss_4 = torch.relu(15 - offset_1).mean() + torch.relu(15 - offset_2).mean()
        loss_5 = loss_5 / len(test_cap)
        loss_sum_weight_list.append(loss_3.data.item())
        loss_random_weight_list.append(loss_2.data.item())
        loss_p.append(loss_4.data.item()) 
        loss_o.append(loss_5.data.item())
        
        print(f"loss_1: {np.array(np.mean(loss_delta_weight_list))}")
        print(f"loss_2: {np.array(np.mean(loss_random_weight_list))}")
        print(f"loss_3: {np.array(np.mean(loss_sum_weight_list))}")
        print(f"loss_4: {np.array(np.mean(loss_p))}")
        print(f"loss_5: {np.array(np.mean(loss_o))}")
        
        for i in range(len(test_cap)):
            cap_sliced, idx = first_slices(test_cap[i])
            label = torch.full((len(cap_sliced),), test_class[i]).to(device)
            t = np.linspace(1, len(test_cap[i]), len(test_cap[i]))
            weight, offset_1, offset_2 = model(torch.tensor(cap_sliced).to(device, non_blocking = True), label)
            top_10 = np.mean(np.partition(np.array(weight.to("cpu").detach()), -10)[-10:])
            print(top_10)
            
            print(offset_1[100])
            print(offset_2[100])
            
            weight = accumulated_sums(np.array(weight.to("cpu").detach()))
            weight_inter = linear_interpolation(t, idx + np.array(offset_2.detach().to("cpu")), weight)
            plt.plot(t, weight_inter, label = "predict",c = "r")
            plt.plot(t, test_weight[i], label = "ground truth", c = "b")
            plt.xlabel("t")
            plt.ylabel("weight")
            # if i == 0 or i == 1:
            #     title = "ibuprofen half filled"
            # elif i == 2 or i == 3:
            #     title = "ibuprofen full filled"
            # elif i == 4 or i == 5:
            #     title = "water half filled"
            # elif i == 6 or i == 7:
            #     title = "water full filled"
            # elif i == 8 or i == 9:
            #     title = "sugar half filled"
            # elif i == 10 or i == 11:
            #     title = "sugar full filled"
            # elif i == 12 or i == 13:
            #     title = "salt half filled"
            # elif i == 14 or i == 15:
            #     title = "salt full filled"
            # elif i == 16 or i == 17:
            #     title = "oil half filled"
            # elif i == 18 or i == 19:
            #     title = "oil full filled"
            # elif i == 20 or i == 21:
            #     title = "vinegar half filled"
            # elif i == 22 or i == 23:
            #     title = "vinegar full filled"
            # elif i == 24 or i == 25:
            #     title = "lentils half filled"
            # elif i == 26 or i == 27:
            #     title = "lentils full filled"
            # elif i == 28 or i == 29:
            #     title = "rice half filled"
            # elif i == 30 or i == 31:
            #     title = "rice full filled"
            # elif i == 32 or i == 33:
            #     title = "ibuprofen half filled(paper)"
            # elif i == 34 or i == 35:
            #     title = "ibuprofen full filled(paper)"
            # elif i == 36 or i == 37:
            #     title = "water half filled(paper)"
            # elif i == 38 or i == 39:
            #     title = "water full filled(paper)"
            # elif i == 40 or i == 41:
            #     title = "sugar half filled(paper)"
            # elif i == 42 or i == 43:
            #     title = "sugar full filled(paper)"
            # elif i == 44 or i == 45:
            #     title = "salt half filled(paper)"
            # elif i == 46 or i == 47:
            #     title = "salf full filled(paper)"
                
            
            if i == 0 or i == 1:
                title = "ibuprofen"
            elif i == 2 or i == 3:
                title = "ibuprofen"
            elif i == 4 or i == 5:
                title = "water"
            elif i == 6 or i == 7:
                title = "water"
            elif i == 8 or i == 9:
                title = "sugar"
            elif i == 10 or i == 11:
                title = "sugar"
            elif i == 12 or i == 13:
                title = "salt"
            elif i == 14 or i == 15:
                title = "salt"
            elif i == 16 or i == 17:
                title = "oil"
            elif i == 18 or i == 19:
                title = "oil"
            elif i == 20 or i == 21:
                title = "vinegar"
            elif i == 22 or i == 23:
                title = "vinegar"
            elif i == 24 or i == 25:
                title = "lentils"
            elif i == 26 or i == 27:
                title = "lentils"
            elif i == 28 or i == 29:
                title = "rice"
            elif i == 30 or i == 31:
                title = "rice"
            elif i == 32 or i == 33:
                title = "ibuprofen half filled(paper)"
            elif i == 34 or i == 35:
                title = "ibuprofen full filled(paper)"
            elif i == 36 or i == 37:
                title = "water half filled(paper)"
            elif i == 38 or i == 39:
                title = "water full filled(paper)"
            elif i == 40 or i == 41:
                title = "sugar half filled(paper)"
            elif i == 42 or i == 43:
                title = "sugar full filled(paper)"
            elif i == 44 or i == 45:
                title = "salt half filled(paper)"
            elif i == 46 or i == 47:
                title = "salf full filled(paper)"
            
            
            
            
            
            
            
            plt.title(f"{title}")
            plt.legend()
            plt.show()
            
    
        
        






