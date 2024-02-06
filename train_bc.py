# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:42:04 2023

@author: yexin
"""

import torch.nn as nn
import time
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
from model import MLP_bc
from dataloader import data_train_bc, data_test_bc
import torch
from progressbar import ProgressBar
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import random
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_dir', type=str, default='C:/Users/yexin/Desktop/liquid/', help='Please change to your experiment path')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-2, help='weight decay')
parser.add_argument('--num_warmup_steps', type=int, default=100, help='num_warmup_steps')
parser.add_argument('--epoch', type=int, default=100, help='The time steps you want to subsample the dataset to,100')
parser.add_argument('--train_continue', type=bool, default= False, help='Set true if continue to train')
parser.add_argument('--test', type = bool, default = True, help = "running on test set")
parser.add_argument('--train', type = bool, default = False, help = "running on train set")
args = parser.parse_args()



if not os.path.exists(args.exp_dir + 'ckpts'):
    os.makedirs(args.exp_dir + 'ckpts')
    
    
    
use_gpu = True
device = 'cuda'


def get_class(label, nums):
    classes = []
    for i in range(len(nums)):
        classes.append(torch.tensor(label[nums[i]]))
        

    return torch.stack(classes).to(device)


if args.train:

    train_cap = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\cap_train_bc.pkl", "rb"))
    train_b = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\b_train_bc.pkl", "rb"))
    
    train_weight_num = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\weight_num_train_bc.pkl", "rb"))
    train_num = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\num_train_bc.pkl", "rb"))
    
    train_dataset = data_train_bc(train_cap, train_b)
    print("finish load training data")
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0)

    

    val_cap = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\cap_test_bc.pkl", "rb"))
    val_b = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\b_test_bc.pkl", "rb"))
    
    val_weight_num = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\weight_num_test_bc.pkl", "rb"))
    val_num = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\num_test_bc.pkl", "rb"))
    
    val_dataset = data_test_bc(val_cap, val_b)
    print("finish load val data")
    
    
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,shuffle=False, num_workers=0)

if args.test:
    test_cap = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\cap_test_bc.pkl", "rb"))
    test_b = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\b_test_bc.pkl", "rb"))
    
    test_weight_num = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\weight_num_test_bc.pkl", "rb"))
    test_num = pickle.load(open(r"C:\Users\yexin\Desktop\liquid\training data\num_test_bc.pkl", "rb"))
    
    test_dataset = data_test_bc(test_cap, test_b)
    print("finish load test data")
    
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False, num_workers=0)

if __name__ == '__main__':
    if args.train:
        np.random.seed(0)
        torch.manual_seed(99)
    
        model = MLP_bc()

        model.to(device)
        if args.train_continue:
            checkpoint = torch.load(r"C:\Users\yexin\Desktop\liquid\ckpts\best_bc.path.tar")
            model.load_state_dict(checkpoint['model_state_dict'])
    
        
    
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
        scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, len(train_dataloader) * args.epoch)
        criterion = nn.BCELoss()
        relu = nn.ReLU()
    
    
        best_val = np.inf
        for epoch in tqdm(range(args.epoch)):
    
            loss_f = []
    
            
            
            print ('here')
            
            bar = ProgressBar(max_value=len(train_dataloader))
    
            for i_batch, sample_batched in bar(enumerate(train_dataloader, 0)):
                model.train(not args.test)
    
    
    
                
                cap = sample_batched[0].to(device, non_blocking = True)
                num = sample_batched[1].to(device, non_blocking = True)
                act = sample_batched[2].to(device, non_blocking = True)
                
                
    
    
                
    
                with torch.set_grad_enabled(not args.test):
                    
                    label = get_class(train_num, num)
                    weight = get_class(train_weight_num, num)
                    act_out = model(cap, label, weight)
                    
                        
                    
                    

                    
                    loss = criterion(act_out.float(), act.float())
                    
                    if not args.test:
    
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                        
                loss_f.append(loss.data.item())
      
                
                if i_batch % 50 ==0 and i_batch > 0:
                    print(f"loss_f: {np.array(np.mean(loss_f))}")

                    
                    
                    
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': np.array(loss.data.item()),},
                     args.exp_dir + 'ckpts/' + str(epoch) + '.path.tar')
                    
                    
                
                
                
                
            val_loss_f = []
            correct_predictions = 0
            total_samples = 0
  
            bar = ProgressBar(max_value=len(val_dataloader))     
            for i_batch, sample_batched in bar(enumerate(val_dataloader, 0)):
                model.eval()
    
    
    
                
                cap = sample_batched[0].to(device, non_blocking = True)
                num = sample_batched[1].to(device, non_blocking = True)
                act = sample_batched[2].to(device, non_blocking = True)
                
                
    
    
                
    
                with torch.set_grad_enabled(False):
                    
                    label = get_class(val_num, num)
                    weight = get_class(val_weight_num, num)
                    act_out = model(cap, label, weight)
                    
                        
                    
                    


                    
                    loss = criterion(act_out.float(), act.float())
    
                    
                        
                        
                val_loss_f.append(loss.data.item())
                
                predicted_labels = (act_out > 0.5).float()

                correct_predictions += (predicted_labels == act.float()).sum().item()
                total_samples += act.size(0)
                
            print(f"final accuracy = {correct_predictions / total_samples}")
                
            print(f"val loss: {np.array(np.mean(val_loss_f))}")
    
            if np.array(np.mean(val_loss_f)) < best_val:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),},
                 args.exp_dir + 'ckpts/' + 'best_bc' + '.path.tar')
                    
                best_val = np.array(np.mean(val_loss_f))
            


    elif args.test:
        np.random.seed(0)
        torch.manual_seed(99)
    
        model = MLP_bc()
        model.to(device)
        checkpoint = torch.load(r"C:\Users\yexin\Desktop\liquid\ckpts\best_bc.path.tar")
        model.load_state_dict(checkpoint['model_state_dict'])
    

        criterion = nn.BCELoss()
        relu = nn.ReLU()
         




        val_loss_f = []
        correct_predictions = 0
        total_samples = 0
        
        print ('here')
        
        bar = ProgressBar(max_value=len(test_dataloader))

        for i_batch, sample_batched in bar(enumerate(test_dataloader, 0)):
            model.eval()



            
            cap = sample_batched[0].to(device, non_blocking = True)
            num = sample_batched[1].to(device, non_blocking = True)
            act = sample_batched[2].to(device, non_blocking = True)
            
            


            

            with torch.set_grad_enabled(False):
                
                label = get_class(test_num, num)
                weight = get_class(test_weight_num, num)
                act_out = model(cap, label, weight)
                
                    
                
                


                
                loss = criterion(act_out.float(), act.float())
                
                

                    
                    
            val_loss_f.append(loss.data.item())
            predicted_labels = (act_out > 0.5).float()

            correct_predictions += (predicted_labels == act.float()).sum().item()
            total_samples += act.size(0)
            
            
            
        print(f"final accuracy = {correct_predictions / total_samples}")
            
        print(f"test loss: {np.array(np.mean(val_loss_f))}")

        
        
        






