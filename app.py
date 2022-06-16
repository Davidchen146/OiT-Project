import io
import json
import os

from flask import Flask, jsonify, request
import torch

from oit_helpers2 import LSTM


app = Flask(__name__)

def load_trained_model(model):
    """ Load state dict from trained model
    """
    save_path = "./base_lstm.pth"
    model.load_state_dict(torch.load(save_path))

def init_model():
    """ Initialize empty model
    """
    hidden_layer_size = 100
    n_layers = 2
    input_size = 1
    output_size = 1

    model = LSTM(input_size=input_size, 
                   hidden_layer_size=hidden_layer_size, 
                   n_layers=n_layers, 
                   output_size=output_size)
    return model    

def gen_prediction(input_tensor):
    model = init_model()
    load_trained_model(model)
    with torch.no_grad():
        model.eval()
        return model(input_tensor)

def format_input(input_data):
    """Takes input data and returns tensor for model prediction.
    
    Args:
        input_data (list) List of lists where each sublist is of len num_lags.
                          Each sublist should be an an array of type float
                          of length num_lags (since model predicts 1 price 
                          from num_lags).
                          
    Output:
        (3D tensor) Model expects 3D tensor
    
    """
    return torch.Tensor(input_data).float().unsqueeze(2)

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'POST to the /predict endpoint using input'})

@app.route('/predict', methods=['POST'])
def predict():
    """Given input, convert to tensor and generate predictions from model.
    
    Note:
        - User supplies input data (should be list of lists of length n_lags)
            - e.g., [[0.,...,6.],[7.,...,13.]]
            - Here, model will generate/return 2 prices, one per sub list
        - `format_input()` converts to tensor and reshapes to (# samples, n_lags, 1)
        - `gen_predicts()` takes tensor and generates/returns predictions 
    """
    if request.method == 'POST':
        data = request.get_json()['input_data']
        ts = format_input(data)
        preds = gen_prediction(ts)
        output = {'output':preds.tolist()}
        return jsonify(output)

if __name__ == '__main__':
    app.run(debug=False, port=1234)