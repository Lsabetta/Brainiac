"""
Author: lakj
"""
import sys
import math
from typing import Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

import torch

# ----- MEASURES -----
# Original Extreme Learning Machine implementation
class ELM:
    def __init__(self, inp, hid, out, device):
        """
        inp: int, size of the input vector
        hid: int, number of the hidden units
        out: int, number of output classes
        device: str, gpu or cpu
        returns: None
        """
        # Could be non-orthogonal too
        self.w = torch.nn.init.orthogonal_(torch.empty(inp, hid)).to(device)
        self.b = torch.rand(1, hid).to(device)
        self.beta = torch.rand(hid, out).to(device)

    def _forward(self, x):
        """
        x: tensor, the input data
        returns: tensor, output scores
        """
        return torch.relu((x @ self.w) + self.b)

    def fit(self, x, y):
        """
        x: tensor, the training data
        returns: None
        """
        # y must be one hot encoded
        self.beta = torch.pinverse(self._forward(x)) @ y

    def predict(self, x):
        """
        x: tensor, the test data
        returns: None
        """
        return self._forward(x) @ self.beta



# ----- MEASURES -----

def accuracy(tp, tn, fp, fn, p, n):
    return (tp+tn)/(p+n)


def precision(tp, tn, fp, fn, p, n):
    if (tp+fp) != 0:
        return tp/(tp+fp)
    else:
        return 0


def recall(tp, tn, fp, fn, p, n):
    if (tp+fn) != 0:
        return tp/(tp+fn)
    else:
        return 0


def f1(tp, tn, fp, fn, p, n):
    denom = 2*tp + fp + fn
    if denom != 0:
        return 2*tp/denom
    else:
        return 0


# ----- TENSOR WRAPPERS -----
class RunningTensor:
    """ Store a tensor and expose an update method to compute a running
    mean item of the stored tensor. this is not complianto with backprop
    because we use deepcopy
    """

    def __init__(self, name: str="empty") -> None:
        """ Requires a name to be uniquely identifiead and a starting tensor """
        self.name = name
        self.n = 0


    def update(self, new_tensor: torch.tensor) -> torch.tensor:
        """ Updates the cumulative tensor with a new instance and compute the new
        mean item.
        """
        if self.n == 0:
            self.n += 1
            self.cum_tensor = copy.deepcopy(new_tensor)
            self.cur_avg = self.cum_tensor / self.n
            return self.cur_avg

        elif new_tensor.dtype != self.cum_tensor.dtype:
            raise TypeError(f"Found tensor of type {new_tensor.dtype} expected tensor of type {self.cum_tensor.dtype}")

        elif new_tensor.shape != self.cum_tensor.shape:
            raise TypeError(f"Found tensor of shape {new_tensor.shape} expected tensor of type {self.cum_tensor.shape}")

        else:
            self.n += 1
            self.cum_tensor += new_tensor
            self.cur_avg = self.cum_tensor / self.n
            return self.cur_avg



class RunningAvg:
    def __init__(self, name: str, num: float) -> None:
        """ Requires a name to be uniquely identifiead and a starting number """
        self.name = name
        self.cum_num = num
        self.n = 1

    def update(self, new_num: float) -> float:
        """ Updates the cumulative num with a new instance and compute the new
        mean number.
        """
        self.n += 1
        self.cum_num += new_num
        return self.cum_num / self.n

# ----- TENSOR MANIPULATIONS -----

def shuffle_tensor(t: torch.tensor) -> torch.tensor:
    """ Given a tensor, it shuffles the rows """
    permuted_idxs = torch.randperm(t.shape[0])
    return t[permuted_idxs]


def shuffle_pair_tensors(t1: torch.tensor, t2: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """ Given two tensors with the same number of rows,
    it shuffles both consistently.  Usually it is used to shuffle images
    and its corresponding labels of a batch
    """

    if t1.shape[0] == t2.shape[0]:
        permuted_idxs = torch.randperm(t1.shape[0])
    else:
        raise ValueError(f"t1 and t2 must have same .shape[0]. Found {t1.shape[0]} and {t2.shape[0]}")

    return t1[permuted_idxs], t2[permuted_idxs]


def sample_k_rows(t: torch.tensor, k:int) -> torch.tensor:
    """ Given a tensor and a integer it choose k random rows
    from the tensor and returns it
    """
    if k <= t.shape[0]:
        permuted_idxs = torch.randperm(t.shape[0])[:k]
    else:
        raise ValueError(f"k must be smaller than tensor.shape[0], Found {k} instead <= {t.shape[0]}")

    return t[permuted_idxs]


def sample_k_rows_pair_tensors(t1: torch.tensor, t2: torch.tensor, k:int) -> Tuple[torch.tensor, torch.tensor]:
    """ Given a pair of tensors and a integer it choose k random rows
    from the tensors (same rows) and returns the tuple
    """
    if k <= t1.shape[0]:
        permuted_idxs = torch.randperm(t1.shape[0])[:k]
    else:
        raise ValueError(f"k must be smaller than tensor.shape[0], Found {k} instead <= {t1.shape[0]}")

    return t1[permuted_idxs], t2[permuted_idxs]

# ----- VISUALIZATION -----
def viz(imgs: torch.tensor, lbls: torch.tensor, filename:str='./default.png', cols: int = 7, show:bool=False) -> bool:
    """ Visualizes a pair of imgs and labels tensors"""

    n = imgs.shape[0]

    rows = math.ceil(n / cols)

    s = 1/(cols/rows)
    if s < 0.3:
        s = 0.3
    #print(rows, cols, s)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15,int(15*s)))
    for i, ax in zip(range(n), axs.flatten()):
        ax.imshow(imgs[i].permute(1,2,0).numpy())
        ax.axis('off')
        ax.set_title(int(lbls[i].item()))

    if show:
        plt.show()
    else:
        plt.savefig(filename)

    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')

    return True


# ----- MODEL DEBUGGING TOOLS -----
def model_info(model):
    """ Prints model info """
    for m in list(model.children()):
        print(m)
        for x in m.parameters():
            print(x.shape, x.requires_grad)
        print('------------------------------------------')

def model_size(model):
    """ Prints model info """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb = (params * 32) / 2**23
    gb = (params * 32) / 2**33
    print(f"Params: {params} MB: {mb:.2f} GB:{gb:.2f} [- 32bit]")
