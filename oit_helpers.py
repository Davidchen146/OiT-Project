"""
TODO: Write docstring
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessors:
    def __init__(self,data):
        self.original_data = data

    def create_timeseries_sequences(self,data, timestep=1):
        """Given timeseries data, convert to timeseries data with given timestep.

        Args:
            data (Pandas.Series): A Pandas series with single input variable
            timestep (int): Timestep to use

        Returns:
            TODO
        """

        if type(data) != pd.Series:
            raise TypeError(f"data must be of type Pandas Series but is instead {type(data)}")

        self.timestep = timestep

        X, Y = [], []
        for i in range(len(data)-timestep):
            X.append(data[i:(i+timestep)])
            Y.append(data[i+timestep])

        return np.array(X), np.array(Y)

    def train_test_split(self,historical,target, prop_train, valid_set):
        """Given data and desired proportion of data to keep for training, split and return train/test data.

        Args:
            data (Pandas.dataframe): A Pandas dataframe
            pct_train (float): Proportion of data to keep for training

        Returns:
            TODO
        """
        n_obs = len(historical)
        train_size = int(n_obs*prop_train)
        X_train,X_test = historical[0:train_size,:], historical[train_size:,:]
        y_train,y_test = target[0:train_size], target[train_size:]
        
        if not valid_set:
            print(f'Training set has shape {X_train.shape}')
            print(f'Test set has shape {X_test.shape}')
        else:
            print(f'Training set has shape {X_train.shape}')
            print(f'Validation set has shape {X_test.shape}')

        return X_train, X_test, y_train, y_test


    def normalize_data(self, X, scaler=None, train=False):
        """Given a list of input features, standardizes them to bring them onto a homogenous scale
        Args:
            X (dataframe): A dataframe of all the input values
                scaler (object, optional): A StandardScaler object. Defaults to None.
            train (bool, optional): If False, means validation set to be loaded and scaler needs to be
                passed to scale it. Defaults to False.
        """
        if train:
            scaler = MinMaxScaler(feature_range=(0,1))
            new_X = scaler.fit_transform(X)
            return (new_X, scaler)
        else:
            new_X = scaler.transform(X)
            return (new_X, None)


class createDataLoader(torch.utils.data.Dataset):
    """This class is the dataset class which is used to load data for training the LSTM 
    to forecast timeseries data
    """

    def __init__(self, inputs, outputs):
        """Initialize the class with instance variables
        Args:
            inputs ([list]): [A list of tuples representing input parameters]
            outputs ([list]): [A list of floats for the closing price]
        """
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        """Returns the total number of samples in the dataset
        """
        return len(self.outputs)
    
    def __getitem__(self, idx):
        """Given an index, it retrieves the input and output corresponding to that index and returns the same
        Args:
            idx ([int]): [An integer representing a position in the samples]
        """
        x = torch.FloatTensor(self.inputs[idx])
        y = torch.FloatTensor([self.outputs[idx]])
        
        return (x, y)

def getDataLoader(x, y, params):
    """Given the inputs, labels and dataloader parameters, returns a pytorch dataloader
    Args:
        x ([list]): [inputs list]
        y ([list]): [target variable list]
        params ([dict]): [Parameters pertaining to dataloader eg. batch size]
    """
    training_set = createDataLoader(x, y)
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    return training_generator
>>>>>>> f05191d8e72bd73e95b6a9ca0bfc5426d3492270


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, 
                 output_size, n_layers=1, do = .05, 
                 device = "cpu"):
        """Initialize the network architecture
        Args:
            input_size ([int]): [Number of time lags to look at for current prediction]
            hidden_layer_size ([int]): [The dimension of RNN output]
            n_layers (int, optional): [Number of stacked RNN layers]. Defaults to 1.
            output_size (int, optional): [Dimension of output].
            do (float, optional): [Dropout for regularization]. Defaults to .05.
        """
        
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.n_layers = n_layers
        self.dropout = do
        self.device = device
        
        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_layer_size,
                            num_layers = n_layers,
                            dropout = do)

        self.linear = nn.Linear(in_features = hidden_layer_size, 
                                out_features = output_size)

        # initialize hidden state with zeros
        # tuple of hidden and cell state
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        """Define the forward propogation logic here
        Args:
            input_seq ([Tensor]): [A 3-dimensional float tensor containing parameters]
        """
        
        #lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        #predictions = self.linear(lstm_out.view(len(input_seq), -1))
        #return predictions[-1]
        
        bs = input_seq.shape[1]
        out, _ = self.lstm(input_seq, self.hidden_cell)
        out = out.contiguous().view(-1, self.hidden_layer_size)
        out = self.linear(out)
        
        return out
        
    
    def predict(self, input):
        """Makes prediction for the set of inputs provided and returns the same
        Args:
            input (torch.Tensor): A tensor of inputs
        """
        
        with torch.no_grad():
            predictions = self.forward(input)
            
        return predictions
            
