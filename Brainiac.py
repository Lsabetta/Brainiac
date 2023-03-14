import torch
import torch.nn as nn

from utils import get_model
import timm
from utils import RunningTensor
from opt import OPT
from torch.utils.data import DataLoader

class Brainiac():
    def __init__(self, model_name, distance_type) -> None:
        #super(Brainiac, self).__init__()
        self.model_name = model_name
        self.centroids = {}
        self.sigmas = {}
        self.covariance_matrix = torch.zeros((10,10))
        self.model, self.preprocessing = get_model(model_name)
        self.model.eval()
        self.head = torch.tensor([])
        self.distance_type = distance_type

    def predict(self, x):
        self.embeddings = self.model.encode_image(x) #store latest embeddings computed
        distances = self.distance()
        prediction = torch.mode(torch.argmin(distances, dim = 1)).values.item()
        return prediction, distances

    def store_new_class(self, label):
        rt = RunningTensor()
        _ = rt.update(self.embeddings.mean(dim = 0).unsqueeze(0))
        self.centroids[label] = rt
        self.sigmas[label] = self.embeddings.std(dim = 0)

    
    def update_class(self, label):
        
        emb_mean = self.embeddings.mean(dim = 0)
        centroid = self.centroids[label].cur_avg
        self.centroids[label].update(emb_mean.unsqueeze(0))
        n = self.centroids[label].n
        
        self.sigmas[label] = (self.sigmas[label]*(n-1) + self.embeddings.std(dim = 0))/n

        '''
        if label not in self.sigmas:
            self.sigmas[label] = torch.cat((centroid, emb_mean.unsqueeze(0))).std(dim = 0)
        else:
            self.sigmas[label] = torch.sqrt((n-1)/n * self.sigmas[label]**2 + 1./n * (emb_mean - centroid.squeeze(0))**2)
        '''

    def distance(self):
        if self.distance_type == "l2":
            return torch.cdist(self.embeddings, torch.cat([c.cur_avg for c in self.centroids.values()], dim = 0))

        if self.distance_type == "l1":
            return torch.cdist(self.embeddings, torch.cat([c.cur_avg for c in self.centroids.values()], dim = 0), p = 1)
        
        if self.distance_type == "normalized_l2":
            distances = torch.tensor([]).to(OPT.DEVICE)
            for i, key in enumerate(self.centroids):
                #if i == 0:
                #    distances = ((self.embeddings - self.centroids[key].cur_avg)/self.sigmas[key]).mean(dim = -1).unsqueeze()
                to_append = (torch.abs(self.embeddings - self.centroids[key].cur_avg)/self.sigmas[key]).mean(dim = -1).unsqueeze(-1)
                distances = torch.cat((distances, to_append), dim =1)
            return distances


    def forward_example(self, x):
        
        for video in DataLoader(x, batch_size=x.shape[0]):
            self.embeddings = self.model.encode_image(video.to(OPT.DEVICE))
            break