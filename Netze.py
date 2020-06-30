import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import operator
import pandas as pd
import numpy as np
import pprint
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import pandasql as sqldf

class Netz(nn.Module):
    def __init__(self):
        super(Netz,self).__init__()
        #self.fc1 = nn.Linear(53, 150)
        self.fc1 = nn.Linear(52, 150)
        self.fc2 = nn.Linear(150, 180)
        self.fc3 = nn.Linear(180, 190)
        self.fc4 = nn.Linear(190, 120)
        self.fc5 = nn.Linear(120, 100)
        self.fc6 = nn.Linear(100, 70)
        self.fc7 = nn.Linear(70, 30)
        self.fc8 = nn.Linear(30, 1)
        self.dropout = nn.Dropout()
        
        
    def forward(self,x):
        x = self.fc1(x.float())
        x = F.relu(x.float())
        x = self.fc2(x.float())
        x = F.relu(x.float())
        x = self.dropout(x)
        x = self.fc3(x.float())
        x = F.relu(x.float())
        x = self.fc4(x.float())
        x = F.relu(x.float())
        x = self.fc5(x.float())
        x = F.relu(x.float())
        x = self.fc6(x.float())
        x = F.relu(x.float())
        x = self.fc7(x.float())
        x = F.relu(x.float())
        x = self.fc8(x.float())
        return x

class NetzDynamic(nn.Module):
    
    def __init__(self, layer_information):
        super(NetzDynamic,self).__init__()
        self.__layer_information = layer_information
        #definieren von einer Moduleliste, die alle Layer des Netzes enthalten wird
        self.layers = nn.ModuleList()
        
        #erzeugen der in layer_information übergebeben Layer und hinzufügen zu layers Moduleliste
        for specs in self.__layer_information.values():
            type_ = specs[0]
            in_ = specs[1]
            out_ = specs[2]
            if type_ == 'linear':
                self.layers.append(nn.Linear(in_,out_))
            if type_ == 'dropout':
                self.layers.append(nn.Dropout())
        
    def forward(self, x):
        #definieren eines dynamischen forward pass, activation function richtet sich nach dictionary key
        lay_idx = 0
        for activation, specs in self.__layer_information.items():
            if activation.startswith('relu'):
                layer = self.layers[lay_idx] #auswählen des relevanten Layers aus layers ModuleList
                if specs[0] == 'dropout':
                    x = F.relu(x.float())#anwenden der activation function
                    x = layer(x)
                else:
                    x = F.relu(x.float())
                    x = layer(x.float()) 
            if activation.startswith('sigmoid'):
                layer = self.layers[lay_idx]
                if specs[0] == 'dropout':
                    x = torch.sigmoid(x.float())
                    x = layer(x)
                else:
                    x = torch.sigmoid(x.float())
                    x = layer(x.float())
                    
            if activation.startswith('tanh'):
                layer = self.layers[lay_idx]
                if specs[0] == 'dropout':
                    x = torch.tanh(x.float())
                    x = layer(x)
                else:
                    x = torch.tanh(x.float())
                    x = layer(x.float())
            if activation.startswith("first"):
                layer = self.layers[lay_idx]
                if specs[0] == 'dropout':
                    x = layer(x)
                else:
                    x = layer(x.float())
            if activation.startswith("no_activation"):
                layer = self.layers[lay_idx]
                if specs[0] == 'dropout':
                    x = layer(x)
                else:
                    x = layer(x.float()) 
            lay_idx += 1#updaten der layerid, um forward pass ins nächste layer zu machen
        
        return x

