"""
TODO: Write docstring
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

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
            
        self.timeseries_sequence_data = {'historical':X,
                                         'target':Y}
        return np.array(X), np.array(Y)
    
    def train_test_split(self,historical,target, prop_train):
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
        print(f'Training set has shape {X_train.shape}')
        print(f'Test set has shape {X_test.shape}')
        self.train_test_data = {'training':{'historical':X_train,
                                            'target':y_train},
                                'test':{'historical':X_test,
                                        'target':y_test}}
        return X_train, X_test, y_train, y_test
        
    
    def normalize_data(self):
        pass


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]