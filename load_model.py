import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score
import pdb
import os
from tst import Transformer
from tst.loss import OZELoss
from torchsummary import summary
from src.dataset_eeg import EEGDataset 
from src.utils import compute_loss, fit, Logger, kfold
from src.benchmark import LSTM, BiGRU, ConvGru, FFN
from src.metrics import MSE
import random
np.random.seed(42)
torch.manual_seed(42)

path =  '/media/kingformatty/easystore/C247/project/transformer/exp/wrong_test_crop_size/conformer_lr_dc0.01_4conv_kernel_channel_last_crop800_normalize/models/best_model26.pth' 


BATCH_SIZE = 4 #set batch size
NUM_WORKERS = 0 #set num of workers, can be ignored
LR = 1e-2 #set learning rate
weight_decay = 1e-2 #set l2 weight decay
EPOCHS = 50 #set training epoch
d_model = 56  #set transformer hidden dimension
q = 8  # Query size
v = 8  # Value size
h = 8  # Number of heads
N = 4  # Number of encoder and decoder to stack
attention_size = 12  # Attention window size
dropout = 0.3  # Dropout rate
pe = 'regular'  # Positional encoding
chunk_mode = None #can be ignored
only_encoder = True #only use transformer encoder or use both encoder and decoder
d_input = 22	  # From dataset, size of input feature
d_output = 4  # From dataset, num of class
embedding_option = 'last' #mean / last #statistical pooling strategy
crop = 1000 #crop from start to crop
conformer = True # whether use conformer based transformer or pure transformer: conformer suggested
normalize = True
conv_dimen = 'channel' #conformer convolution dimension, 'feature'/ 'channel':
                       #'feature' option does convolution across sequence dimension (1D)
                       #'channel' option does convolution across feature and sequence dimension

device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")
net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe,only_encoder = only_encoder,embedding_option = embedding_option, conformer = conformer, conv_dimen = conv_dimen)
                  
net.load_state_dict(torch.load(path))
net.eval()	
dataset_test = EEGDataset(train = False,normalize = normalize, crop = crop)
dataloader_test = DataLoader(dataset_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS
                             )
predictions = torch.empty(len(dataloader_test.dataset))
idx_prediction = 0
y_true = torch.empty(len(dataloader_test.dataset))
for x, y in tqdm(dataloader_test, total=len(dataloader_test)):
    netout = net(x.to(device)).cpu()
        #pdb.set_trace()
    netout = torch.argmax(netout, dim = 1)
    predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
    y_true[idx_prediction: idx_prediction+ x.shape[0]] = y
    idx_prediction += x.shape[0]
    
y_pred = predictions.numpy()
y_true = y_true.numpy()

correct = np.sum(np.where(y_pred == y_true, 1, 0))
accuracy = correct / y_true.shape[0]
print(accuracy)
