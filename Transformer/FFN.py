import torch
from torch import nn 

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output, **kwargs) -> None:
        super().__init__(**kwargs)
        # ffn_num_input = (attention) hidden_size 
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        # ffn_num_output = (attention) hidden_size
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
    
""" class ffn(nn.Module):
    def __init__(self, ffn_input, ffn_hidden, hidden_size) -> None:
        super().__init__()
        self.layer1 = nn.Linear(ffn_input, ffn_hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ffn_hidden, hidden_size)
    
    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x))) """
    