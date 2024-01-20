import torch
from torch import nn 
import EncoderDecoder


class AttentionDecoder(EncoderDecoder.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self): #用来画图
        raise NotImplementedError