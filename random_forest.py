# %% Libraries
from math import exp
from sys import path
import os
import datetime
from numpy.core import numeric
from numpy.core.fromnumeric import mean
import pandas as pd 
import numpy as np
import sklearn 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = {}
model = {}
# %%



# %% Is the dataset something we can make into a class?
class DataSet():
    def __init__(self, path, label_names=['model'], ohe_names=['transmission','fuelType'], random_state=420):
        self.df = pd.read_csv(path)
        self.random_state = random_state
        self.encodings = LabelEncoder() 
        self.scaler = StandardScaler()
        self.label_names = label_names
        self.ohe_labels = ohe_names
        self.colnames = self.df.columns

    def __len__(self):
        return(len(df))

    def show_sample(self):
        print(df.head())

    def encode(self):
        
        self.df[self.label_names] = self.encodings.fit_transform(self.df[self.label_names])
        self.df = pd.concat([self.df, pd.get_dummies(self.df[self.ohe_labels])], axis = 1)
        print('Labels Encoded')
    
    def inverse_encode(self):
        self.df[self.label_names] = self.encodings.inverse_transform(self.df[self.label_names]) 
        print('Labels Decoded')

    def give_splits(self):
        return(train_test_split(self.df.drop('price',axis=1),self.df.price, test_size=0.25, random_state = self.random_state))


    def reset(self):
        self.df = pd.read_csv(path)

    def transforms(self):
        self.df = self.df.drop(self.ohe_labels, axis=1)
        self.df['year'] = datetime.datetime.today().year - self.df['year']

# %%

# %%
def import_datasets():
    data = {}
    for file in os.listdir('/home/kahlil/Documents/PythonPlayground/cars_pricing/data'):
        name = file[:-4]
        path = 'data/'+file
        data[name] = DataSet(path)
    print('Data Imported')
    return(data)
    
def initialize_models():
    for dataset in data:
        modelname = str(dataset)
        model[modelname] = RandomForestRegressor()

def train_models():
    for m in model:
        data[m].encode()
        data[m].transforms()
        X_train, X_test, Y_train, Y_test = data[m].give_splits()
        model[m].fit(X_train, Y_train)

        print(m)

def show_scores():
    for m  in model:
        X_train, X_test, Y_train, Y_test = data[m].give_splits()
        accuracy = (model[m].score(X_test, Y_test))
        print('Model for {} has an accuracy of {}'.format(m, accuracy))
# %%
# %%


# %%
