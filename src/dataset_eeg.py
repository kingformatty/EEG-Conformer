#copyright
#Author: Jinhan Wang
#Year: 2021

import json
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
import torch
import pdb



class EEGDataset(Dataset):
    def __init__(self, train = True,normalize = False, crop = 1000):
        #super().__init__()
        if train == False:
            X = np.load('/media/kingformatty/easystore/C247/project/data/X_test.npy') #(443, 22, 1000)
            self.y = np.load('/media/kingformatty/easystore/C247/project/data/y_test.npy') #(443,)
            self.person = np.load('/media/kingformatty/easystore/C247/project/data/person_test.npy') #(443, 1)
        else:
            X = np.load('/media/kingformatty/easystore/C247/project/data/X_train_valid.npy') # (2115, 22, 1000)
            self.y = np.load('/media/kingformatty/easystore/C247/project/data/y_train_valid.npy') # (2115,)
            self.person = np.load('/media/kingformatty/easystore/C247/project/data/person_train_valid.npy') # (2115, 1)

        if crop:
            #X = X[:,:,:crop]
            X = X[:,:,:crop]
            print(X.shape)
            
        
        
        self.n_samples = X.shape[0]
        #standardize

        #pdb.set_trace()
        if normalize:
            mean = np.mean(X, axis = 0) #(22, 1000)
            std = np.std(X, axis = 0) # (22, 1000)
            self.X_norm = (X - mean ) / std
            self.X_norm = X
        else:
            self.X_norm = X

        #transpose
        self.X_norm = np.transpose(self.X_norm,(0,2,1))
        #pdb.set_trace()
        
        #formulate label into 0 to 3
        for i in range(len(self.y)):
            if self.y[i] == 769:
                self.y[i] = 0
            elif self.y[i] == 770:
                self.y[i] = 1
            elif self.y[i] == 771:
                self.y[i] = 2
            elif self.y[i] == 772:
                self.y[i] = 3
                
        #define crop
        #pdb.set_trace()
            

    def __len__(self):
        return self.n_samples
    def __getitem__(self, i):
        #pdb.set_trace()
	
        return (torch.from_numpy(np.array(self.X_norm[i])).to(torch.float32), torch.tensor(torch.from_numpy(np.array(self.y[i])),dtype = torch.long))
