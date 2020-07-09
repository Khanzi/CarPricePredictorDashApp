#%% Libraries   
import torch 
from torch import nn, sqrt
import numpy as np 
import pandas as pd 
import seaborn as sns
from torch.autograd import Variable
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Linear 
from torch.autograd import Variable
from sklearn import preprocessing
from torch.nn.modules.loss import MSELoss
from torch.optim import optimizer
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import mean_squared_error

# %% Importing Data
toyota = pd.read_csv('data/toyota.csv')
toyota.head()
#%% Encoding the vars
#x_toyota = toyota.drop('price', axis=1)
x_toyota = toyota.apply(preprocessing.LabelEncoder().fit_transform)
x_toyota
# %%
class CarsData(Dataset):
    def __init__(self,path):
        df = pd.read_csv(path, header = None)
        df = df.apply(preprocessing.LabelEncoder().fit_transform)
        self.X = df.drop(df.columns[2], axis=1).values.astype('float32')
        self.Y = df.values[:,2].astype('float32')

        self.Y = self.Y.reshape((len(self.Y),1))

    def __len__(self):
        return(len(self.X))

    def __getitem__(self, idx):
        return(self.X[idx], self.Y[idx])

    def get_splits(self, n_test = 0.25):
        test_size = round(n_test * len(self.X))
        train_size = (len(self.X) - test_size)
        return torch.utils.data.random_split(self, [train_size, test_size])




# %%
class CarsModel(nn.Module):
    def __init__(self, n_inputs):
        super(CarsModel, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 4)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()

        self.hidden2 = nn.Linear(4,2)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()

        self.hidden3 = nn.Linear(2,1)
        nn.init.xavier_uniform_(self.hidden3.weight)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        return(x)

# %%
def prep_data(path):
    dataset = CarsData(path)

    train, test = dataset.get_splits()

    train_dl = DataLoader(train, batch_size=32, shuffle = True)
    test_dl = DataLoader(test, batch_size = 128, shuffle=False)
    return(train_dl, test_dl)
# %%
def train_model(train_dl, model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()


# %%
def eval_model(test_dl, model):
    predictions = list()
    actuals = list()

    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()

        actual = targets.numpy()
        actual = actual.reshape((len(actual),1))

        predictions.append(yhat)
        actuals.append(actual)
        
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)

        mse = mean_squared_error(actuals, predictions)
        return(mse)

#%%
def predict(row, model):
    row = torch.Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return(yhat)
# %%
path = 'data/toyota.csv'
train_dl, test_dl = prep_data(path)
model = CarsModel(8)
train_model(train_dl, model)

# %%
eval_model(test_dl, model)




# %%
predict([6,18,1,3504,3,21,16,8], model) # Actual is 1451

# %%

predict([14,13,1,5061,3,6,39,3], model) # Actual is 88

# %%
np.sqrt(eval_model(test_dl,model))

# %%
