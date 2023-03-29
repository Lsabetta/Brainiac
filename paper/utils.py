import copy
import torch
import timm
import clip
import open_clip
from opt import OPT
import random
import numpy as np
from torchvision import datasets
from read_core50 import AllCore50Dataset

class RunningTensor:
    """ Store a tensor and expose an update method to compute a running
    mean item of the stored tensor. this is not complianto with backprop
    because we use deepcopy
    """

    def __init__(self) -> None:
        self.n = 0

    def update(self, new_tensor: torch.tensor) -> torch.tensor:
        """ Updates the cumulative tensor with a new instance and compute the new
        mean item.
        """
        if self.n == 0:
            self.n += 1
            self.cum_tensor = copy.deepcopy(new_tensor)
            self.cur_avg = self.cum_tensor 
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





def get_model(model_name):

    if model_name == "CLIP":
        model, preprocessing = clip.load("ViT-B/32", device=OPT.DEVICE)
    elif model_name == "openCLIP":
        model, _, preprocessing = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=OPT.DEVICE)
    else:
        model = timm.create_model(model_name, num_classes = 0, pretrained=True)
        preprocessing = lambda x: x
    return model, preprocessing


def check_known(cls, distances, threshold):
    if distances.mean(dim = 0)[cls] < threshold:
        return True
    else:
        return False 

            
def set_seeds(seed):
    """ Set reproducibility seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False                                                                                                                            
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False            


def get_dataset(dset_name, transform):
    if dset_name == 'CIFAR100':
        return datasets.CIFAR100(root=OPT.DATA_PATH, train=True, download=True, transform=transform)
    elif dset_name == 'CIFAR10':
        return datasets.CIFAR10(root=OPT.DATA_PATH, train=True, download=True, transform=transform)
    elif dset_name == 'Flowers102':
        return datasets.Flowers102(root=OPT.DATA_PATH, split='train', download=True, transform=transform)
    elif dset_name == 'INaturlist':
        return datasets.INaturalist(root=OPT.DATA_PATH, version='2017', download=True, transform=transform)
    elif dset_name == 'LFWPeople': 
        return datasets.LFWPeople(root=OPT.DATA_PATH, split='10fold', download=True, transform=transform)
    elif dset_name == 'FGVCAircraft': 
        return datasets.FGVCAircraft(root=OPT.DATA_PATH, train=True, download=True, transform=transform)
    elif dset_name == 'CelebA':
        return datasets.CelebA(root=OPT.DATA_PATH, split='train', target_type="identity", download=True, transform=transform)
    elif dset_name == 'CORE50':
        return AllCore50Dataset(OPT.DATA_PATH, finegrane=False, transform=transform)
    else:
        raise ValueError(f"Dataset {dset_name} not supported")
