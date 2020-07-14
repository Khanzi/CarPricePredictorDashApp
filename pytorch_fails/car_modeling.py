# 
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
    """Neural network with 3 layers

    """
    def __init__(self, n_inputs):
        super(CarsModel, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 4) # Hidden layers with 4 outputs    
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()

        self.hidden2 = nn.Linear(4,2)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()

        self.hidden3 = nn.Linear(2,1)
        nn.init.xavier_uniform_(self.hidden3.weight)

    def forward(self, x):
        """Forward pass the neural net

        Args:
            x (torch.Tensor): X tensor to be trained
        """
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        return(x)

# %%
def prep_data(path):
    """Creates two data loaders for test and training data

    Args:
        path ([str]): file path to .csv
    """
    dataset = CarsData(path) # Creates a CarsData class with path

    train, test = dataset.get_splits() # Creates splits 

    # Creates pytorch data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle = True)
    test_dl = DataLoader(test, batch_size = 128, shuffle=False)
    return(train_dl, test_dl)
# %%
def train_model(train_dl, model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

    for epoch in range(10000):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()


# %%
def eval_model(test_dl, model):
    """ Runs and evaluates the model on the test dataset

    Args:
        test_dl ([torch.utils.data.DataLoader]): Test data loader
        model ([type]): Model to be evaluated
    """
    # Initializing predictions and actuals lists; used for calculations
    predictions = list()
    actuals = list()

    for i, (inputs, targets) in enumerate(test_dl): # Enum through test data
        yhat = model(inputs) # performs test data prediction
        yhat = yhat.detach().numpy() # Converts tensor pred into python number

        actual = targets.numpy() # Converts tensor to python number 
        actual = actual.reshape((len(actual),1)) # Changed dim to 1

        predictions.append(yhat) # Appends predictions to list
        actuals.append(actual) # Appends actuals to list
        
        predictions = np.vstack(predictions) # Stacks predictions
        actuals = np.vstack(actuals) # Stacks actuals   

        mse = mean_squared_error(actuals, predictions) # Calculates MSE
        return(mse, np.sqrt(mse)) # Returns mse and rmse   

#%%
def predict(row, model):
    """Returns a prediction for a given row of data.
    Note: Data must have the same number of features as model

    Args:
        row ([list]): X data for the prediction
        model ([type]): Model for prediction
    """
    row = torch.Tensor([row]) # Converts the list into a torch tensor
    yhat = model(row) # Performs the prediction
    yhat = yhat.detach().numpy() # Converts the prediction to python number
    return(yhat)
