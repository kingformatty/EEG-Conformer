import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
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

# Training parameters



    
DATASET_PATH = '/media/kingformatty/easystore/C247/project/data/'


# Model parameters
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
embedding_option = 'mean' #mean / last #statistical pooling strategy
crop = 1000 #crop from start to crop
conformer = True # whether use conformer based transformer or pure transformer: conformer suggested
conv_dimen = 'channel' #conformer convolution dimension, 'feature'/ 'channel':
                       #'feature' option does convolution across sequence dimension (1D)
                       #'channel' option does convolution across feature and sequence dimension (2D)
                       

#normalize

exp_dir = 'conv_dimension_check'
#exp_dir = 'exp/conformer_lr001_dc001'+'_'+'4_conv_kernel_'+conv_dim+'_'+emb+'_crop'+str(crop_size)+'/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
    os.makedirs(exp_dir+'models')
if os.path.exists(exp_dir + 'log'):
    os.remove(exp_dir+'log')
 
# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Load dataset
dataset_train_val = EEGDataset(train = True,normalize = True, crop = crop)

# Split between train, validation and test
#dataset_train, dataset_val= random_split(dataset_train_val,[1692,423])
dataset_train, dataset_val= random_split(dataset_train_val,[1692,423])
#pdb.set_trace()
dataloader_train = DataLoader(dataset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=False
                              )

dataloader_val = DataLoader(dataset_val,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS
                            )
dataset_test = EEGDataset(train = False,normalize = True, crop = crop)
dataloader_test = DataLoader(dataset_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS
                             )

# Load transformer with Adam optimizer and MSE loss function
                #N = conformer_num
net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe,only_encoder = only_encoder,embedding_option = embedding_option, conformer = conformer, conv_dimen = conv_dimen).to(device)
                #pdb.set_trace()  
                #summary(net, (800,22))
optimizer = optim.SGD(net.parameters(), lr=LR,weight_decay = weight_decay)
loss_function = nn.CrossEntropyLoss()
net.train()
# Fit model     
with tqdm(total=EPOCHS) as pbar:
                
    loss, val_acc_best, train_acc_best,model_path= fit(net, optimizer, loss_function, dataloader_train,
                           dataloader_val, dataloader_test, epochs=EPOCHS, pbar=pbar, device=device,exp_dir = exp_dir)

    device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")
    net_best = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe,only_encoder = only_encoder,embedding_option = emb, conformer = conformer, conv_dimen = conv_dimen)
    net_best.load_state_dict(torch.load(model_path))
    net.eval()
    predictions = torch.empty(len(dataloader_test.dataset))
    idx_prediction = 0
    y_true = torch.empty(len(dataloader_test.dataset))
    with torch.no_grad():
        for x, y in tqdm(dataloader_test, total=len(dataloader_test)):
            netout = net_best(x.to(device)).cpu()
        #pdb.set_trace()
            netout = torch.argmax(netout, dim = 1)
            predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
            y_true[idx_prediction: idx_prediction+ x.shape[0]] = y
            idx_prediction += x.shape[0]
        
#pdb.set_trace()

#evaluation

            y_pred = predictions.numpy()
            y_true = y_true.numpy()

            correct = np.sum(np.where(y_pred == y_true, 1, 0))
            accuracy = correct / y_true.shape[0]
#pdb.set_trace()
            print('Test Accuracy for model with best validation performance is {}'.format(accuracy))
            with open(exp_dir+'VAL_Test_Accuracy','w') as f:
                f.write('Test Accuracy ')
                f.write(str(accuracy))
                f.write('\n')
                f.write('Best Val Accuracy ')
                f.write(str(val_acc_best))
                f.write('Best Train Accuracy ')
                f.write(str(train_acc_best))
                f.write('\n')

