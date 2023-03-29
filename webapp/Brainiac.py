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
        #super(Brainiac, self).__init__()
        self.model_name = model_name
        self.centroids = {}
        self.all_embeddings = {}
        self.sigmas = {}
        self.covariance_matrix = torch.zeros((10,10))
        self.model, self.preprocessing = get_model(model_name)
        self.model.eval()
        self.head = torch.tensor([])
        self.distance_type = distance_type
        self.index_2_label = {} #this is a variable that map indices to labels
        self.label_2_index = {} #this is a variable that map labels to indices
        self.prediction = None
        self.known = None
        self.distances = None
        self.cls_image_examples = {}

    def predict(self, x):
        self.embeddings = self.model.encode_image(x)
        distances = self.distance()
        prediction = torch.mode(torch.argmin(distances, dim = 1)).values.item()
        return prediction, distances

    def store_new_class(self, label, image):
        if label not in self.cls_image_examples:
            self.cls_image_examples[label] = image
        self.all_embeddings[label] = self.embeddings.detach()
        self.centroids[label] = self.embeddings.mean(dim = 0).detach()
        self.sigmas[label] = self.embeddings.std(dim = 0)
        self.index_2_label[len(self.index_2_label)] = label
        self.label_2_index[label] = len(self.label_2_index)


    
    def update_class(self, label):
        DESTINY = random.uniform(0,1)
        if DESTINY < OPT.UPDATE_PROBABILITY:
            emb_mean = self.embeddings.mean(dim = 0).unsqueeze(0).detach()
            self.all_embeddings[label] = torch.cat((self.all_embeddings[label], self.embeddings.detach()))
            self.centroids[label] = self.all_embeddings[label].mean(dim=0)
            self.sigmas[label] = self.all_embeddings[label].std(dim=0)

        #n = self.centroids[label].n
        
        #self.sigmas[label] = (self.sigmas[label]*(n-1) + self.embeddings.std(dim = 0))/n

        '''
        if label not in self.sigmas:
            self.sigmas[label] = torch.cat((centroid, emb_mean.unsqueeze(0))).std(dim = 0)
        else:
            self.sigmas[label] = torch.sqrt((n-1)/n * self.sigmas[label]**2 + 1./n * (emb_mean - centroid.squeeze(0))**2)
        '''

    def distance(self):
        if self.distance_type == "l2":
            return torch.cdist(self.embeddings.to(torch.float32), torch.stack([c.to(torch.float32) for c in self.centroids.values()]))

        if self.distance_type == "l1":
            return torch.cdist(self.embeddings.to(torch.float32), torch.stack([c.to(torch.float32) for c in self.centroids.values()]), p = 1)

        if self.distance_type == "inverse_cosine":
            a = self.embeddings.to(torch.float32)
            b = torch.stack([c.to(torch.float32) for c in self.centroids.values()])
            norm = torch.sqrt(((a*a).sum(dim = 1).unsqueeze(1))@((b*b).sum(dim = 1).unsqueeze(1).T))
            return 1. - torch.divide((a@b.T), norm)
            
        if self.distance_type == "normalized_l2":
            distances = torch.tensor([]).to(OPT.DEVICE)
            for i, key in enumerate(self.centroids):
                #if i == 0:
                #    distances = ((self.embeddings - self.centroids[key].cur_avg)/self.sigmas[key]).mean(dim = -1).unsqueeze()
                to_append = (torch.abs(self.embeddings - self.centroids[key])/self.sigmas[key]).mean(dim = -1).unsqueeze(-1)
                distances = torch.cat((distances, to_append), dim =1)
            return distances


    def forward_example(self, x, first_iteration):
        transform = T.ToPILImage()
        b = torch.tensor([])
        if self.preprocessing:
            for i, im in enumerate(x):
                im = transform(im)
                b = torch.cat((b, self.preprocessing(im).unsqueeze(0)))
        print(b.shape)
        for video in DataLoader(b, batch_size=b.shape[0]):
            self.embeddings = self.model.encode_image(video.to(OPT.DEVICE))
            break
        if first_iteration:
            return
        distances = self.distance()
        prediction = torch.mode(torch.argmin(distances, dim = 1)).values.item()
        return prediction, distances