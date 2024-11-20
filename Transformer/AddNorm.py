from torch import nn

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout) -> None:
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    

""" class addnorm(nn.Module):
    def __init__(self, norm_shape, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(normalized_shape=norm_shape)
        
    def forward(self, X, Y):
        return self.layernorm(self.dropout(Y) + X) """