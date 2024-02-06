# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:12:13 2023

@author: yexin
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.init as init
from torchvision.models import resnet18


def prepare_sequence(data, max_length=100):
    # If data is shorter than max_length, pad it
    if len(data) < max_length:
        padding_length = max_length - len(data)
        # Assuming you pad with zeros
        padded_data = torch.cat([data, torch.zeros(padding_length)], dim=0)
        # Create a mask: 1 for real tokens, 0 for padding tokens
        mask = torch.cat([torch.ones(len(data)), torch.zeros(padding_length)], dim=0)
    # If data is longer than max_length, truncate it
    elif len(data) > max_length:
        padded_data = data[:max_length]
        mask = torch.ones(max_length)
    # If data is exactly max_length
    else:
        padded_data = data
        mask = torch.ones(max_length)
    
    return padded_data, mask



class Transformer(nn.Module):
    def __init__(self, d_model = 128, nhead = 4, num_layers = 2, output_dim = 3):
        super(Transformer, self).__init__()
        
        self.device = torch.device("cuda")
        self.projection = nn.Linear(10, d_model)
        
        self.positional_encodings = self._generate_sin_positional_encodings(seq_len=10, d_model=d_model)
        
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model, nhead, dropout = 0.05)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers)

        self.fc = nn.Linear(d_model * 10, output_dim)
        self.relu = nn.ReLU()

    def _generate_sin_positional_encodings(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_encodings = torch.zeros(seq_len, d_model)
        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)
        pos_encodings = pos_encodings.unsqueeze(0).to(self.device)
        return pos_encodings

    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = self.projection(x)
        pos = self.positional_encodings
        x = x + pos

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(batch_size, -1)
        

        

        
        x = self.fc(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        return weight, offset_1, offset_2
    



class MLP(nn.Module):
    def __init__(self, d_model=256, output_dim=2):
        super(MLP, self).__init__()
        
        self.projection = nn.Linear(10 * 20, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(5)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset = x[:, 1]
        
        return weight, offset
    
    
    
    
class CNN1D(nn.Module):
    def __init__(self, input_dim = 10, sequence_len = 10):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv1d(128, 128, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool1d(2)
        
        # Compute the flattened size after convolutions and pooling
        flattened_size = sequence_len
        flattened_size //= 2  
        
        self.fc1 = nn.Linear(128 * flattened_size, 128)
        self.fc2 = nn.Linear(128, 3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Adjust dimensions to fit Conv1D (N, C, L) format
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)

        
        x = self.conv2(x)
        x = self.relu(x)

        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.relu(x)

        
        x = self.conv5(x)
        x = self.relu(x)

        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]

        
        return weight, offset_1, offset_2



class MLP_2(nn.Module):
    def __init__(self, d_model=256, output_dim=3):
        super(MLP_2, self).__init__()
        
        self.projection = nn.Linear(10 * 20, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2
    
    

class MLP_2_10(nn.Module):
    def __init__(self, d_model=256, output_dim=3):
        super(MLP_2_10, self).__init__()
        
        self.projection = nn.Linear(10 * 10, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2


    
    
    
class MLP_simple(nn.Module):
    def __init__(self, d_model=256, output_dim=1):
        super(MLP_simple, self).__init__()
        
        self.projection = nn.Linear(10 * 10, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        x = x[:, 0]

        
        return x
    
    
    
    
    
class MLP_simple_class(nn.Module):
    def __init__(self, d_model=256, output_dim=1, class_size = 2, embedding_dim = 50):
        super(MLP_simple_class, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings = class_size, embedding_dim=embedding_dim)

        self.projection = nn.Linear(10 * 20 + class_size, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, label):
        batch_size, seq_len, shape = x.shape
        label = torch.nn.functional.one_hot(label, num_classes=2)
        
        x = x.reshape(batch_size, -1)
        x = torch.concatenate((x, label), dim = 1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        x = x[:, 0]

        
        return x


class MLP_2_class(nn.Module):
    def __init__(self, d_model=256, output_dim=3, class_size = 2, embedding_dim = 50):
        super(MLP_2_class, self).__init__()
        
        
        self.embedding = nn.Embedding(num_embeddings = class_size, embedding_dim=embedding_dim)
        self.projection = nn.Linear(10 * 20 + class_size, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, label):
        batch_size, seq_len, shape = x.shape
        # label = self.embedding(label)
        label = torch.nn.functional.one_hot(label, num_classes=2)
        
        x = x.reshape(batch_size, -1)
        x = torch.concatenate((x, label), dim = 1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2





class MLP_d(nn.Module):
    def __init__(self, d_model=256, output_dim=10):
        super(MLP_d, self).__init__()
        
        self.projection = nn.Linear(10 * 100, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
            # x = self.dropout(x)
            
        x = self.out(x)


        
        return x
    
    
    
    
    
    
    
class LSTM(nn.Module):
    def __init__(self, input_dim = 10, hidden_dim = 256, num_layers = 10, output_dim = 3):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 10, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to("cuda")
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to("cuda")
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out.reshape(len(out), -1))
        
        weight = out[:, 0]
        offset_1 = out[:, 1]
        offset_2 = out[:, 2]
        
        return weight, offset_1, offset_2  
    
    
    
    
    
    
    
    
    
class MLP_2_resnet(nn.Module):
    def __init__(self, d_model=256, output_dim=3):
        super(MLP_2_resnet, self).__init__()
        
        self.projection = nn.Linear(10 * 20, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.02)
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            a = x
            x = linear(x)
            x += a
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2
    
    
class MLP_2_10_3(nn.Module):
    def __init__(self, d_model=512, output_dim=3):
        super(MLP_2_10_3, self).__init__()
        
        self.projection = nn.Linear(10 * 10, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.02)
        
        for linear in self.linears:
            init.kaiming_normal_(linear.weight)  # He Initialization
            init.zeros_(linear.bias)             # Zero Initialization for Bias
            
        init.xavier_uniform_(self.out.weight)  # Xavier Initialization
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            # a = x
            x = linear(x)
            # x += a
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2
    
class MLP_2_10_33(nn.Module):
    def __init__(self, d_model=1024, output_dim=3):
        super(MLP_2_10_33, self).__init__()
        
        self.projection = nn.Linear(10 * 10, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        
        for linear in self.linears:
            init.kaiming_normal_(linear.weight)  # He Initialization
            init.zeros_(linear.bias)             # Zero Initialization for Bias
            
        init.xavier_uniform_(self.out.weight)  # Xavier Initialization
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            # a = x
            x = linear(x)
            # x += a
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2
    
    
class MLP_2_10_3_tanh(nn.Module):
    def __init__(self, d_model=1024, output_dim=3):
        super(MLP_2_10_3_tanh, self).__init__()
        
        self.projection = nn.Linear(10 * 10, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        
        for linear in self.linears:
            init.kaiming_normal_(linear.weight)  # He Initialization
            init.zeros_(linear.bias)             # Zero Initialization for Bias
            
        init.xavier_uniform_(self.out.weight)  # Xavier Initialization
        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            # a = x
            x = linear(x)
            # x += a
            x = self.relu(x)
            x = self.dropout(x)
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2
   
    
   

    
class MLP_2_10_resnet(nn.Module):
    def __init__(self, d_model=256, output_dim=3):
        super(MLP_2_10_resnet, self).__init__()
        
        self.projection = nn.Linear(10 * 10, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        

        
    def forward(self, x):
        batch_size, seq_len, shape = x.shape
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            a = x
            x = linear(x)
            x = x + a
            x = self.relu(x)
            x = self.dropout(x)
            
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2    
 

  
    
 
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.head_dim = query_dim // num_heads

        assert self.head_dim * num_heads == query_dim, "query_dim must be divisible by num_heads"

        self.query_linear = nn.Linear(query_dim, query_dim)
        self.key_linear = nn.Linear(key_value_dim, query_dim)
        self.value_linear = nn.Linear(key_value_dim, query_dim)
        self.out_linear = nn.Linear(query_dim, query_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # Split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply the attention to the values
        context = torch.matmul(attention_probs, value)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.query_dim)
        output = self.out_linear(context)

        return output    
    
class MLP_2_10_resnet_class(nn.Module):
    def __init__(self, d_model=256, output_dim=3, class_size = 5, embedding_dim = 50):
        super(MLP_2_10_resnet_class, self).__init__()
        
        self.projection = nn.Linear(10 * 10 + embedding_dim, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        # self.cross_attention = CrossAttention(query_dim=d_model, key_value_dim=embedding_dim, num_heads=4)
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        self.embedding = nn.Embedding(num_embeddings = class_size, embedding_dim=embedding_dim)
        
        # for linear in self.linears:
        #     init.kaiming_normal_(linear.weight)  # He Initialization
        #     init.zeros_(linear.bias)             # Zero Initialization for Bias
            
        # init.xavier_uniform_(self.out.weight)  # Xavier Initialization
        
    def forward(self, x, label):
        batch_size, seq_len, shape = x.shape
        label = self.embedding(label)
        # label = torch.nn.functional.one_hot(label, num_classes=3)
        x = x.reshape(batch_size, -1)
        x = torch.concatenate((x, label), dim = 1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            a = x
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x) 
            x += a
            
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2   
    
    
class MLP_2_class_resnet(nn.Module):
    def __init__(self, d_model=256, output_dim=3, class_size = 2, embedding_dim = 50):
        super(MLP_2_class_resnet, self).__init__()
        
        
        self.embedding = nn.Embedding(num_embeddings = class_size, embedding_dim=embedding_dim)
        self.projection = nn.Linear(10 * 20 + class_size, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, label):
        batch_size, seq_len, shape = x.shape
        # label = self.embedding(label)
        label = torch.nn.functional.one_hot(label, num_classes=2)
        
        x = x.reshape(batch_size, -1)
        x = torch.concatenate((x, label), dim = 1)
        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            a = x
            x = linear(x)
            x += a
            x = self.relu(x)
            x = self.dropout(x)
            
            
        x = self.out(x)
        x = self.relu(x)
        weight = x[:, 0]
        offset_1 = x[:, 1]
        offset_2 = x[:, 2]
        
        return weight, offset_1, offset_2    







class MLP_bc(nn.Module):
    def __init__(self, d_model=256, output_dim=1, class_size = 5, weight_size = 4, embedding_dim = 50):
        super(MLP_bc, self).__init__()
        
        self.projection = nn.Linear(10 * 10 + embedding_dim * 2, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        self.embedding1 = nn.Embedding(num_embeddings = class_size, embedding_dim=embedding_dim)
        self.embedding2 = nn.Embedding(num_embeddings = weight_size, embedding_dim=embedding_dim)
        
        self.sig = nn.Sigmoid()

        
    def forward(self, x, label, weight):
        batch_size, seq_len, shape = x.shape
        label = self.embedding1(label)
        weight = self.embedding2(weight)
        
        x = x.reshape(batch_size, -1)
        x = torch.concatenate((x, label, weight), dim = 1)

        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            a = x
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x) 
            x += a
            
            
        x = self.out(x)
        x = self.sig(x)
        x = x[:, 0]

        
        return x


class MLP_bc_value(nn.Module):
    def __init__(self, d_model=256, output_dim=1, class_size = 5, weight_size = 4, embedding_dim = 50):
        super(MLP_bc_value, self).__init__()
        
        self.projection = nn.Linear(10 * 10 + embedding_dim * 2, d_model)
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(7)])
        self.out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        self.embedding1 = nn.Embedding(num_embeddings = class_size, embedding_dim=embedding_dim)
        self.linear = nn.Linear(1, embedding_dim)
        
        self.sig = nn.Sigmoid()

        
    def forward(self, x, label, weight):
        batch_size, seq_len, shape = x.shape
        label = self.embedding1(label)
        weight = self.linear(weight.reshape(-1, 1).float())
        
        x = x.reshape(batch_size, -1)
        x = torch.concatenate((x, label, weight), dim = 1)

        x = self.projection(x)
        x = self.relu(x)
        
        for linear in self.linears:
            a = x
            x = linear(x)
            x = self.relu(x)
            x = self.dropout(x) 
            x += a
            
            
        x = self.out(x)
        x = self.sig(x)
        x = x[:, 0]
        
        return x

    
    
    
    
