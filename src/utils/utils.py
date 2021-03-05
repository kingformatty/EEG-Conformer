import csv

import torch
import numpy as np
import pdb

def compute_loss(net: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 device: torch.device = 'cpu') -> torch.Tensor:
    """Compute the loss of a network on a given dataset.

    Does not compute gradient.

    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.

    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss = 0
    predictions = torch.empty(len(dataloader.dataset))
    idx_prediction = 0
    y_true = torch.empty(len(dataloader.dataset))
    with torch.no_grad():
        for x, y in dataloader:
            netout = net(x.to(device)).cpu()
            #pdb.set_trace()
            running_loss += loss_function(netout,y)
            netout = torch.argmax(netout, dim = 1)
            predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
            y_true[idx_prediction: idx_prediction+ x.shape[0]] = y
            idx_prediction += x.shape[0]
    y_pred = predictions.numpy()
    y_true = y_true.numpy()

    correct = np.sum(np.where(y_pred == y_true, 1, 0))
    accuracy = correct / y_true.shape[0]
    return running_loss / len(dataloader), accuracy
