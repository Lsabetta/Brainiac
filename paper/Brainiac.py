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
        self.centroids = {}
        self.squared_centroids = {}
        self.covariance_matrices = {}
        self.inverted_covariance_matrices = {}
        self.ns = {}
        self.sigmas = {}
        
        # We use this dictionary to map the real classes
        # with the model's classes names
        self.index_2_label = {}
        self.label_2_index = {}
        
        self.last_prediction = None


    def predict(self, x, iteration):
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
        distances = self.distance(embeddings, iteration, distance_type=self.distance_type)
        self.last_prediction = torch.mode(torch.argmin(distances, dim = 1)).values.item()
        return self.last_prediction, embeddings, distances 


    def store_new_centroid(self, embeddings, label):
        """
        Connects the real label with the internal label representation of the Brainiac
        """
        self.centroids[label] = embeddings.mean(dim = 0).detach().to('cpu')
        self.squared_centroids[label] = (self.centroids[label]**2).to(OPT.DEVICE)#.detach().to('cpu')
        self.sigmas[label] = torch.ones((OPT.EMBEDDING_SIZE[OPT.MODEL])).detach().to(OPT.DEVICE)#.to('cpu')
        self.covariance_matrices[label] = torch.zeros((OPT.EMBEDDING_SIZE[OPT.MODEL], OPT.EMBEDDING_SIZE[OPT.MODEL])).detach().to('cpu')
        self.ns[label] = 1

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

        self.ns[label] += 1
        
        self.centroids[label] = (self.centroids[label].to(OPT.DEVICE)*(self.ns[label] - 1) + embeddings.mean(dim = 0))/self.ns[label]

        self.squared_centroids[label] = (self.squared_centroids[label]*(self.ns[label] - 1) + embeddings.mean(dim = 0).detach()**2)/self.ns[label]
        self.sigmas[label] = torch.sqrt((self.squared_centroids[label] - self.centroids[label]**2 + 10e-5)*self.ns[label]/(self.ns[label]-1))
        
        
        #self.covariance_matrices[label] = ((self.ns[label]-1)*self.covariance_matrices[label] + torch.outer((embeddings.squeeze(0)-self.centroids[label]), (embeddings.squeeze(0)-self.centroids[label])))/(self.ns[label])
      
        #self.inverted_covariance_matrices[label] = torch.inverse((1-10e-4)* self.covariance_matrices[label] + 10e-4*torch.eye(OPT.EMBEDDING_SIZE[OPT.MODEL]).to(OPT.DEVICE))
    

    def distance(self, embeddings, iteration, distance_type = "inverse_cosine"):
        """
        Computes the distance between embeddings and centroids, based on the distance, type

        Args:
            embeddings, torch.tensor(processing_frames, 512)
        
        Returns:
            distances: torch.tensor(1,len(centroids))
        """
        # L2 distance
        if distance_type == "l2":
            return torch.cdist(embeddings.to(torch.float32), torch.stack([c.to(torch.float32).to(OPT.DEVICE) for c in self.centroids.values()]))

        # L1 distance
        if distance_type == "l1":
            return torch.cdist(embeddings.to(torch.float32), torch.stack([c.to(torch.float32) for c in self.centroids.values()]), p = 1)

        # Cosine similarity
        if distance_type == "inverse_cosine":
            a = embeddings.to(torch.float32)
            b = torch.stack([c.to(torch.float32).to(OPT.DEVICE) for c in self.centroids.values()])
            norm = torch.sqrt(((a*a).sum(dim = 1).unsqueeze(1))@((b*b).sum(dim = 1).unsqueeze(1).T))
            return 1. - torch.divide((a@b.T), norm)
        
        # Normalized, needs sigmas of all 512 dimension of all each centroids
        if distance_type == "normalized_l2":
            distances = torch.tensor([]).to(OPT.DEVICE)
            for i, label in enumerate(self.centroids):
                numerator = (embeddings - self.centroids[label].to(OPT.DEVICE))**2
                denominator = self.sigmas[label].to(OPT.DEVICE)**2
                to_append = torch.sqrt((numerator/denominator).mean(dim = -1)).unsqueeze(-1)
                distances = torch.cat((distances, to_append), dim =1)
            return distances

        if distance_type == "mix_l2":
            if iteration > 500:
                OPT.THRESHOLD = 0.6
                distances = torch.tensor([]).to(OPT.DEVICE)
                for i, label in enumerate(self.centroids):
                    numerator = (embeddings - self.centroids[label].to(OPT.DEVICE))**2
                    denominator = self.sigmas[label].to(OPT.DEVICE)**2
                    to_append = torch.sqrt((numerator/denominator).mean(dim = -1)).unsqueeze(-1)
                    distances = torch.cat((distances, to_append), dim =1)
                return distances
                
            else: 
                OPT.THRESHOLD = 18
                return torch.cdist(embeddings.to(torch.float32), torch.stack([c.to(torch.float32).to(OPT.DEVICE) for c in self.centroids.values()]))
        
        if distance_type == "mahalanobis":
            distances = torch.tensor([]).to(OPT.DEVICE)
            for i, label in enumerate(self.centroids):
                if self.ns[label] == 1:
                    distances = torch.cat((distances, torch.ones((1,1)).to(OPT.DEVICE)), dim = 1)
                else:
                    to_append = (embeddings - self.centroids[label]).mean(dim = 0)@self.inverted_covariance_matrices[label]@(embeddings - self.centroids[label]).mean(dim = 0)
                    distances = torch.cat((distances, to_append.unsqueeze(0).unsqueeze(0)), dim = 1)
            return distances

        raise(f'Distance {self.distance_type} not implemented')
