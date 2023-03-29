import torch
import torch.nn as nn

from utils import get_model
import timm
from utils import RunningTensor
from opt import OPT
from torch.utils.data import DataLoader
import torchvision.transforms as T
import random


class Brainiac():
    def __init__(self, model_name, distance_type) -> None:
        
        # Defines the model
        self.model_name = model_name
        self.model, self.preprocessing = get_model(model_name)
        self.model.eval()
        self.distance_type = distance_type

        # We use this to compute centroids and all sigmas
        # we store everything just for comodity...
        self.all_embeddings = {}
        self.centroids = {}
        self.sigmas = {}
        
        # We use this dictionary to map the real classes
        # with the model's classes names
        self.index_2_label = {}
        self.label_2_index = {}
        
        self.last_prediction = None


    def predict(self, x):
        """ 
        Return prediction and distances with respect to the centroids 
        The prediction is calculated as the mode calculated over the batch
        
        Args:
            x, torch.tensor(processing_frames, 3, 224, 224)
        
        Return:
            prediction, int
            distances, torch.tensor(processing_frames, len(centroids))
        """
        embeddings = self.model.encode_image(x)
        distances = self.distance(embeddings)
        self.last_prediction = torch.mode(torch.argmin(distances, dim = 1)).values.item()
        return self.last_prediction, embeddings, distances 


    def store_new_centroid(self, embeddings, label):
        """
        Connects the real label with the internal label representation of the Brainiac
        """
        self.all_embeddings[label] = embeddings.detach()
        self.centroids[label] = embeddings.mean(dim = 0).detach()
        self.sigmas[label] = embeddings.std(dim = 0)
        self.index_2_label[len(self.index_2_label)] = label
        self.label_2_index[label] = len(self.label_2_index)


    def update_centroid(self, embeddings, gt_label, probability=1., self_learning = False):
        """
        Brainiac knows this class and updates its representation, with a given probability 
        """
        if random.uniform(0,1) < probability:
            label = gt_label
        else:
            if self_learning:
                label = self.index_2_label[self.last_prediction]
            else:
                return
        self.all_embeddings[label] = torch.cat((self.all_embeddings[label], embeddings.detach()))
        self.centroids[label] = self.all_embeddings[label].mean(dim=0)
        self.sigmas[label] = self.all_embeddings[label].std(dim=0)


    def distance(self, embeddings):
        """
        Computes the distance between embeddings and centroids, based on the distance, type

        Args:
            embeddings, torch.tensor(processing_frames, 512)
        
        Returns:
            distances: torch.tensor(1,len(centroids))
        """
        # L2 distance
        if self.distance_type == "l2":
            return torch.cdist(embeddings.to(torch.float32), torch.stack([c.to(torch.float32) for c in self.centroids.values()]))

        # L1 distance
        if self.distance_type == "l1":
            return torch.cdist(embeddings.to(torch.float32), torch.stack([c.to(torch.float32) for c in self.centroids.values()]), p = 1)

        # Cosine similarity
        if self.distance_type == "inverse_cosine":
            a = embeddings.to(torch.float32)
            b = torch.stack([c.to(torch.float32) for c in self.centroids.values()])
            norm = torch.sqrt(((a*a).sum(dim = 1).unsqueeze(1))@((b*b).sum(dim = 1).unsqueeze(1).T))
            return 1. - torch.divide((a@b.T), norm)
        
        # Normalized, needs sigmas of all 512 dimension of all each centroids
        if self.distance_type == "normalized_l2":
            distances = torch.tensor([]).to(OPT.DEVICE)
            for i, key in enumerate(self.centroids):
                to_append = (torch.abs(embeddings - self.centroids[key])/self.sigmas[key]).mean(dim = -1).unsqueeze(-1)
                distances = torch.cat((distances, to_append), dim =1)
            return distances

        raise(f'Distance {self.distance_type} not implemented')
