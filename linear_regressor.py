# %% Libraries
from sys import path
from numpy.core import numeric
import pandas as pd 
import numpy as np
import sklearn 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
d = defaultdict(LabelEncoder)
# %%



# %% Is the dataset something we can make into a class?
class DataSet():
    def __init__(self, path, random_state=420):
        self.df = pd.read_csv(path)
        self.random_state = random_state
        self.encodings = defaultdict(LabelEncoder)
        self.encoding_names = self.df.select_dtypes(include=object).columns

    def __len__(self):
        return(len(df))

    def show_sample(self):
        print(df.head())

    def encode(self):
        self.df[self.encoding_names] = self.df[self.encoding_names].apply(lambda x: d[x.name].fit_transform(x))
        print('Labels Encoded')
        print(self.df.head(3))
    
    def inverse_encode(self):
        self.df[self.encoding_names] = self.df[self.encoding_names].apply(lambda x: d[x.name].inverse_transform(x))
        print('Labels Decoded')
        print(self.df.head(3))

    def give_splits(self):
        return(train_test_split(self.df.drop('price',axis=1),self.df.price, test_size=0.25, random_state = self.random_state))

# %%
toyota = DataSet('data/toyota.csv')
toyota.encode()
X_train, X_test, Y_train, Y_test = toyota.give_splits()
# %%
model = DecisionTreeRegressor()

# %%
model.fit(X_train, Y_train)

# %%
y_preds = model.predict(X_test)
np.sqrt(mean_squared_error(Y_test, y_preds))



