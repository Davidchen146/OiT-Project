"""
TODO: Write docstring
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessors:
    def __init__(self, data):
        self.original_data = data
        
    def ensure_equal_timeseries(self,min_date,max_date,data=None):
        if data is None:
            data = self.original_data
        dates = pd.date_range(min_date, max_date, freq="D")
        df1 = pd.DataFrame(index=dates)
        df_bc = df1.join(data,how='left')
        
        n_missing = df_bc.Close.isnull().sum()
        print(f"Results in {n_missing} missing elements")
        if n_missing > 0:
            print("Imputing missing elements using forward fill method")
            df_bc = df_bc.fillna(method='ffill')
                                 
        return df_bc
        

    def create_timeseries_sequences(self,data, timestep=1):
        """Given timeseries data, convert to timeseries data with given timestep.

        Args:
            data (Pandas.Series): A Pandas series with single input variable
            timestep (int): Timestep to use

        Returns:
            TODO
        """

        self.timestep = timestep

        data_raw = data.values # convert to numpy array
        data_ = []

        # create all possible sequences of length look_back
        for i in range(len(data_raw) - timestep): 
            data_.append(data_raw[i:(i + timestep)])
        
        _data = np.array(data_)
        return _data

    def train_test_split(self, lagged_data, prop_test):
        """Given data and desired proportion of data to keep for training, split and return train/test data.

        Args:
            data (Pandas.dataframe): A Pandas dataframe
            pct_train (float): Proportion of data to keep for training

        Returns:
            TODO
        """
        test_set_size = int(np.round(prop_test*lagged_data.shape[0]));
        train_set_size = lagged_data.shape[0] - (test_set_size);

        x_train = lagged_data[:train_set_size,:-1,:]
        y_train = lagged_data[:train_set_size,-1,:]

        x_test = lagged_data[train_set_size:,:-1]
        y_test = lagged_data[train_set_size:,-1,:]
        
        print('x_train.shape = ',x_train.shape)
        print('y_train.shape = ',y_train.shape)
        print('x_test.shape = ',x_test.shape)
        print('y_test.shape = ',y_test.shape)

        return [x_train, y_train, x_test, y_test]

    def normalize_data(self, X, scaler=None, train=False):
        """Given a list of input features, standardizes them to bring them onto a homogenous scale
        Args:
            X (dataframe): A dataframe of all the input values
                scaler (object, optional): A StandardScaler object. Defaults to None.
            train (bool, optional): If False, means validation set to be loaded and scaler needs to be
                passed to scale it. Defaults to False.
        """
        if train:
            # If we're training, make a new MinMaxScaler with default range. -1 is min, 1 is max
            scaler = MinMaxScaler(feature_range=(-1,1))
            # fit_transform calculates the min and max values of the data scales the data according to it
            new_X = scaler.fit_transform(X)
            return (new_X, scaler)
        else:
            # If we're validating or testing, transform the data using parameters determined beforehand
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
    training_generator = torch.utils.data.DataLoader(training_set, drop_last=True, **params)
    return training_generator

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, 
                 n_layers, output_size, do=0):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_layer_size = hidden_layer_size

        # Number of hidden layers
        self.n_layers = n_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, n_layers, batch_first=True)

        # Readout layer
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
    def init_hidden(self, batch_size):
        """Initialize hidden layers
        
        Args:
            batch_size (int):
        """
        
        return torch.zeros(self.n_layers,batch_size,self.hidden_layer_size).requires_grad_()
        
    def forward(self, input_seq):
        # Initialize hidden state with zeros
        h0 = self.init_hidden(input_seq.size(0))

        # Initialize cell state
        c0 = self.init_hidden(input_seq.size(0))

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(input_seq, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.linear(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
    def predict(self, input_seq):
        """Makes prediction for the set of inputs provided and returns the same
        Args:
            input (torch.Tensor): A tensor of inputs
        """
        
        with torch.no_grad():
            predictions = self.forward(input_seq)
            
        return predictions