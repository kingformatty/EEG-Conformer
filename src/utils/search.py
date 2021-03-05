import csv
from pathlib import Path
import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
import pdb
from src.utils.utils import compute_loss

def fit(net, optimizer, loss_function, dataloader_train, dataloader_val, dataloader_test,epochs=10, pbar=None, device='cpu',exp_dir = None):
    val_loss_best = np.inf
    val_acc_best = 0
    test_acc_best = 0
    train_acc_best = 0
    # Prepare loss history
    for idx_epoch in range(epochs):
        predictions = torch.empty(len(dataloader_train.dataset))
        idx_prediction = 0
        y_true = torch.empty(len(dataloader_train.dataset))
        for idx_batch, (x, y) in enumerate(dataloader_train):
            
            # Propagate input
            netout = net(x.to(device))

            # Comupte loss
            #pdb.set_trace()
            #netout (batch,4)
            #y (batch)
            #for i in range(len(y)):
            #pdb.set_trace()
            #factor = 0.1
            loss = loss_function(netout,y.to(device))
            #l1_penalty = torch.nn.L1Loss(size_average=False)
            #reg_loss = 0
            #for param in net.parameters():
            #    reg_loss += l1_penalty(param, torch.zeros(param.shape).to(device))
            #loss += factor * reg_loss
            optimizer.zero_grad()
            # Backpropage loss
            loss.backward()
            #pdb.set_trace()
            # Update weights
            optimizer.step()
            with torch.no_grad():
            #pdb.set_trace()
                netout = torch.argmax(netout, dim = 1)
                predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
                y_true[idx_prediction: idx_prediction+ x.shape[0]] = y
                idx_prediction += x.shape[0]
            if idx_batch % 20 == 0:
                print('Batch {} done, loss {}'.format(idx_batch, loss.item())) 
                
            with open(exp_dir+'log','a') as f:
                f.write('Batch {} done, loss {}'.format(idx_batch, loss.item()))
                f.write('\n')
        
        # Training Evaluation
        y_pred = predictions.numpy()
        y_true = y_true.numpy()

        correct = np.sum(np.where(y_pred == y_true, 1, 0))
        accuracy = correct / y_true.shape[0]
        if accuracy >= train_acc_best:
            train_acc_best = accuracy
        #validation Evaluation
        val_loss, val_acc = compute_loss(net, dataloader_val, loss_function, device)
        
        
        #save model
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            model_name = 'best_model'+str(idx_epoch)+'.pth'
            torch.save(net.state_dict(), exp_dir +'models/'+ model_name)
            model_path = exp_dir + 'models/'+ model_name
            
        if pbar is not None:
            pbar.update()
        with open(exp_dir + 'log','a') as f:
            f.write('epoch {} done, train loss: {}, train acc: {}, val loss: {}, val acc: {}, best val acc {}'.format(idx_epoch, loss.item(), accuracy, val_loss.item(), val_acc, val_acc_best))
            f.write('\n')
        print('epoch {} done, train loss: {}, train acc: {}, val loss: {}, val acc: {}, best val acc {}'.format(idx_epoch, loss.item(), accuracy, val_loss.item(), val_acc, val_acc_best))
    
    return val_loss_best, val_acc_best, train_acc_best, model_path



def kfold(dataset, n_chunk, batch_size, num_workers):    
    indexes = np.arange(len(dataset))
    chunks_idx = np.array_split(indexes, n_chunk)

    for idx_val, chunk_val in enumerate(chunks_idx):
        chunk_train = np.concatenate([chunk_train for idx_train, chunk_train in enumerate(chunks_idx) if idx_train != idx_val])
        
        subset_train = Subset(dataset, chunk_train)
        subset_val = Subset(dataset, chunk_val)
        
        dataloader_train = DataLoader(subset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers
                             )
        dataloader_val = DataLoader(subset_val,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers
                             )
        
        yield dataloader_train, dataloader_val

def leargnin_curve(dataset, n_part, validation_split, batch_size, num_workers):
    # Split train and val
    val_split = int(len(dataset) * validation_split)
    subset_train, subset_val = random_split(dataset, [len(dataset) - val_split, val_split])

    dataloader_val = DataLoader(subset_val,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers
                                 )

    for idx in np.linspace(0, len(subset_train), n_part+1).astype(int)[1:]:
        subset_learning = Subset(dataset, subset_train.indices[:idx])
        dataloader_train = DataLoader(subset_learning,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers
                                 )

        yield dataloader_train, dataloader_val

class Logger:
    def __init__(self, csv_path, model_name='undefined', params=[]):
        csv_path = Path(csv_path)

        if csv_path.is_file():
            self.csv_file = open(csv_path, 'a')
            self.writer = csv.DictWriter(self.csv_file, ['date', 'model'] + params)
        else:
            self.csv_file = open(csv_path, 'w')
            self.writer = csv.DictWriter(self.csv_file, ['date', 'model'] + params)
            self.writer.writeheader()

        self.model_name = model_name

    def log(self, **kwargs):
        kwargs.update({
            'date': datetime.datetime.now().isoformat(),
            'model': self.model_name
        })
        self.writer.writerow(kwargs)
        self.csv_file.flush()

    def __del__(self):
        self.csv_file.close()
