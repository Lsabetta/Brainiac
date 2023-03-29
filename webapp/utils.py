import copy
import torch
import timm
import cv2
from einops import rearrange
from torchvision import transforms as trs
import clip
import open_clip
from PIL import Image
import torchshow as ts
from opt import OPT
import random
import numpy as np

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

def verdict(cls_idx, known, label_2_index, video):
    if known:
        ts.show(video[0])
        answer = input(f"I know what this is! It's a {label_2_index[cls_idx]}. Do you agree?(y/n)\n")
        if answer in ["y", "yes", "Yes", "YES"]:
            known_for_real = True
        else:
            known_for_real = False

        if known_for_real:
            return known_for_real, label_2_index[cls_idx]
        else:
            real_cls_name = input(f"Oh Dang.. Would you tell me what it is than?\n")
            return known_for_real, real_cls_name
    else:
        ts.show(video[0])
        answer = input(f"The closest guess would be {label_2_index[cls_idx]}, but I think the truth is I never saw this object.\nNew object! Right?(y/n)\n")
        if answer in ["y", "yes", "Yes", "YES"]:
            known_for_real = False
        else:
            known_for_real = True
        if not known_for_real:
            real_cls_name = input("Would you tell me what this is?\n")
            return known_for_real, real_cls_name
        else:
            real_cls_name = input(f"Oh Dang.. Would you tell me what it is than?\n")
            return known_for_real, real_cls_name
            
def set_seeds(seed):
    """ Set reproducibility seeds """
    # torch.manual_seed(seed)                                                                                                                                         
    # random.seed(seed)                                                                                                                                               
    # np.random.seed(seed)                                                                                                                                            

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
        