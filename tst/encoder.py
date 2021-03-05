import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from tst.multiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk, MultiHeadAttentionWindow
from tst.positionwiseFeedForward import PositionwiseFeedForward


class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 conformer = False,
                 conv_dimen = 'feature'):
        """Initialize the Encoder block"""
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)
        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)
        self.conformer = conformer
        self.conv_dimen = conv_dimen
        #single convolution layer
        if self.conv_dimen == 'feature':
            self._conv = nn.Conv1d(in_channels = 56, out_channels = 56,kernel_size = 10, stride = 2, padding = 2)
        elif self.conv_dimen == 'channel':
            self._conv = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (20,1), stride = 1)
        
        #conformer block's convolution
        #self._conv = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (10,1), stride = 1)
        self._elu = nn.ELU()
        #self.bn1 = nn.BatchNorm2d(16)
        #self._conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (10,1), stride = 1)
        #self.bn2 = nn.BatchNorm2d(32)
        #self._conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (10,1), stride = 1)
        #self.bn3 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d((8,1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """        
        #pdb.set_trace()
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        #pdb.set_trace()
        
        #convolution
        #pdb.set_trace()
        #pdb.set_trace()
        #x = x.unsqueeze(1) #append another dimension as convolustion channel
        if self.conformer:
            if self.conv_dimen == 'feature':
                #pdb.set_trace()
                x = torch.transpose(x,1,2)
                x = self._conv(x)
                x = torch.transpose(x,1,2)
            elif self.conv_dimen == 'channel':
                x = x.unsqueeze(1) # (N,C,H,W) = (N,1,1000,56)
                x = self._conv(x) #(N,C_out, H_out, W_out)
                #x = self._elu(x)
                x = self.maxpool(x)
                x = x.view(x.shape[0],-1, x.shape[3])
                
                
        #pdb.set_trace()
        #x = self._elu(x)
        #x = self.bn1(x)
        #x = self.maxpool(x)
        #x = self._conv2(x)
        #x = self._elu(x)
        #x = self.bn2(x)
        #x = self.maxpool(x)
        #x = self._conv3(x)
        #x = self._elu(x)
        #x = self.bn3(x)
        #x = self.maxpool(x)
        #x = self._dopout(x)    
        
        #map back / fallten
        #x = x.view(x.shape[0],-1,56)   
        # Feed forward
        #pdb.set_trace()
            
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)
        #pdb.set_trace()

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map
