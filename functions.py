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


def load_data(directory = 'sliced_data'):
    '''
        Funktion, die die aufbereiteten/vorbereiteten Daten aus existierenden CSV Dateien 
        einliest und je nach Vollständigkeit in zwei Dictionaries abspeichert,
        geschlüsselt nach der jeweiligen RaceId
    '''
    if os.path.exists(directory):
        csv_filenames = []
        #auslesen aller csv file dateinamen aus formula 1 datensatz und abspeichern in liste
        for filename in os.listdir(os.getcwd()+'/'+directory):
            typ = filename.split('.')[-1]
            name = filename.split('.')[0]
            if typ == 'csv':
                csv_filenames.append(filename)
        sliced_races = {}
        #einlesen und abspeichern als dataframe aller dateien
        for file in csv_filenames:
            try:
                df = pd.read_csv(directory+'/'+file, engine = 'python', sep = ';', decimal = '.')
                del df['Unnamed: 0']
            except Exception as e:
                df = pd.read_csv(directory+'/'+file, engine = 'c', sep = ';', decimal = '.')
                del df['Unnamed: 0']
                print(e)
            f = int(file.split('_')[-1].split('.')[0]) #raceid wird als key gesetzt
            sliced_races[f] = df
        print('Einlesen der sliced Dateien erfolgreich')
    else:
        raise ('sliced Dateien können nicht eingelesen werden, da kein entsprechendes Verzeichnis existiert!')
   
    return sliced_races

def train_dev_test(data_dict, train_p = 0.7, dev_p = 0.2, test_p = 0.1, nogo_columns = []):
    
    if round(train_p+dev_p+test_p,1) !=1.0:
        raise ValueError ('No valid train/dev/test distribution')
    
    '''
        Daten werde in einem Dictionary übergeben, Dataframes werden in dieser 
        Funktion geshuffled und dann in einen Traindatensatz, einen Development-
        datensatz und in einen Testdatensatz aufgeteilt.
    '''
    #aufteilen in train, dev, test counter
    train_count = round(len(data_dict.keys())*train_p, 0)
    dev_count = train_count+round(len(data_dict.keys())*dev_p,0)
    test_count = len(data_dict.keys())-(train_count+dev_count)
    
    #shufflen der übergebenen Daten
    keys = list(data_dict.keys())
    random.shuffle(keys)
    data_shuffled = {}
    for key in keys:
        data_shuffled[key] = data_dict[key]
        
    #erzeugen separater train,dev,test dictionaries
    train = {}
    dev = {}
    test = {}
    c = 0
    
    #daten sollen nicht in tensoren umgewandelt werden
    for id, df in data_shuffled.items():
        #entfernen nicht gewollter spalten aus dataframe
        cols = [col for col in df.columns if col not in nogo_columns]
        df = df[cols]
        if c < train_count:
            train[id] = df
        elif c >= train_count and c < dev_count:
            dev[id] = df
        else:
            test[id] = df
        c += 1
                
    return train, dev, test

def to_tensor(train_data, dev_data, test_data, nogo_columns = []):
    '''
        Funktion erhält zuvor erzeugte train, dev und test Dictionarys mit Formel 1 Rennen
        und wandelt die DateFrames in Tensoren um. Den Tensoren wird jeweils ihr Target Value
        (podium_podition) zugeordnet. In Dictionary Form werden die Tensoren zurück gegeben.
        Nicht gewollte Spalten (in nogo_columns) werden entfernt.
    '''
    train = []
    train_ = {}
    dev = []
    dev_ = {}
    test = []
    test_ = {}
    
    for id, race in train_data.items():
        for did in race.driverId.unique():
            temp = race.where(race.driverId == did).dropna(how = "all")
            temp_y = list(temp["podium_position"])
            #nicht gewollte Attribute werden aus Datensatz entfernt (bspw. podium_position)
            cols = [col for col in temp.columns if col not in nogo_columns]
            temp_x = temp[cols]
            #die letze runde eines Fahrers wird betrachtet (enthält kumulierte Informationen über zuvor gefahrene Runden)
            temp_x = temp_x.tail(1)
            x_tensor = torch.tensor(temp_x[cols].values)
            train.append((x_tensor, [temp_y[0]]))
        train_[id] = train
        train = []
    for id, race in dev_data.items():
        for did in race.driverId.unique():
            temp = race.where(race.driverId == did).dropna(how = "all")
            temp_y = list(temp["podium_position"])
            cols = [col for col in temp.columns if col not in nogo_columns]
            temp_x = temp[cols]
            temp_x = temp_x.tail(1)
            x_tensor = torch.tensor(temp_x[temp_x.columns].values)
            dev.append((x_tensor, [temp_y[0]]))
        dev_[id] = dev
        dev = []
    for id, race in test_data.items():
        for did in race.driverId.unique():
            temp = race.where(race.driverId == did).dropna(how = "all")
            temp_y = list(temp["podium_position"])
            cols = [col for col in temp.columns if col not in nogo_columns]
            temp_x = temp[cols]
            temp_x = temp_x.tail(1)
            x_tensor = torch.tensor(temp_x[temp_x.columns].values)
            test.append((x_tensor, [temp_y[0]]))
        test_[id] = test
        test = []
        
    return train_,dev_,test_