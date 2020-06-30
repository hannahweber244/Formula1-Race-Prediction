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

from functions import load_data, to_tensor, train_dev_test
from Netze import Netz, NetzDynamic
from Optimizer import HP_Layer_Optimizer, HP_Optimizer

def start():

    print('Starting Formula 1 Prediction')
    cuda = input('Do you want to use cuda? [y/n]: ')
    print('loading data...')
    sliced_races = load_data(directory = 'sliced_data')
    #definieren nicht gewollter Attribute (Attribute die nicht dem Modell übergeben werden sollen):
    nogo_columns = ['year', 
                'podium_position', 
                'raceId',
                'lap_number',
                'driverId', 
                'driver_fullname',
                'total_milliseconds',
               'lap_in_milliseconds',
               'total_minutes']
    #train dev test splitting
    print('splitting data...')
    train, dev, test = train_dev_test(sliced_races)
    #umwandeln in Tensoren und entfernen der nogo_columns
    print('converting data to tensors...')
    train_T, dev_T, test_T = to_tensor(train, dev, test, nogo_columns)

    print('Train:',len(train))
    print('Dev:',len(dev))
    print('Test:',len(test))
    print('Rennen insgesamt:', len(train)+len(dev)+len(test))
    print(25*'=')
    print('Tensoren:')
    print('Train:',len(train_T))
    print('Dev:',len(dev_T))
    print('Test:',len(test_T))
    print('Rennen insgesamt:', len(train_T)+len(dev_T)+len(test_T))


    print('calling the NN Layer Optimizer...')
    if cuda.lower() == 'y':
        l = HP_Layer_Optimizer(layer_range=(7,12),
                       max_epochs = 2,
                       input_start = 52,
                       activations = ['relu'], 
                       random_activation=True,
                       create_combinations=True, 
                       create_variations = True, 
                       num_variations = 5,
                       pure_activations = True,
                       cuda = True)
    else:
        l = HP_Layer_Optimizer(layer_range=(10,12),
                       max_epochs = 2,
                       input_start = 52,
                       activations = ['relu'], 
                       random_activation=True,
                       create_combinations=True, 
                       create_variations = True, 
                       num_variations = 5,
                       pure_activations = True,
                       cuda = False)
    #l.model_specs_combinations = netze
    l.train_data = train_T
    l.test_data = dev_T
    l.validate_combinations()
    l.get_all_information()
    opt_combination = l.opt_combination

    print('Layer Optimizer finished, starting Epoch and learningrate Optimizer...')
    opt_={}
    for key in opt_combination.keys():
        if key != 'mae':#mae wird noch in dictionary von HP_Optimizer hinzugefügt und muss entfernt werden
            opt_[key] = opt_combination[key]
    #überprüfen ob cuda verwendet werden soll, dann Aufruf des Optimizers für beide Netze
    if cuda.lower() == 'y':
        h = HP_Optimizer(lr_range = [0.0001,0.00045],step_size = 0.0001, max_epochs=(3,5),cuda = True)
        h_dynamic = HP_Optimizer(lr_range = [0.0001,0.00045],step_size = 0.0001, max_epochs=(3,5),cuda = True, dynamic = True, dyn_combination = opt_)
    else:
        h = HP_Optimizer(lr_range = [0.0001,0.00045],step_size = 0.0001, max_epochs=(3,5),cuda = False)
        h_dynamic = HP_Optimizer(lr_range = [0.0001,0.00045],step_size = 0.0001, max_epochs=(3,5),cuda = False, dynamic = True, dyn_combination = opt_)

    h.help()
    #Zuweisen der Trainingsdaten
    h.train_data = train_T
    h_dynamic.train_data = train_T
    #Zuweisen der Dev Daten als Testdaten
    h.test_data = dev_T
    h_dynamic.test_data = dev_T
    #Aufruf der Funktion die verschiedene Kombinationen der Epochen und Lernraten gegeneinander vergleicht
    h.validate_combinations()
    h_dynamic.validate_combinations()
    #Ausgabe der Ergebnisse des Vergleichs aus self.validate_combinations()
    print('Statisch Netz():\n')
    h.get_all_information()
    print('\n',25*'=','\n')
    print('Dynamisches Netz:\n')
    h_dynamic.get_all_information()

    print('Last Optimizer finished')



if __name__ == '__main__':
    start()