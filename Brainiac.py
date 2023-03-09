import torch
import torch.nn as nn

from utils import get_model
import timm
from utils import RunningTensor

class Brainiac():
    def __init__(self, model_name) -> None:
        #super(Brainiac, self).__init__()
        self.model_name = model_name
        self.centroids = {}
        self.model, self.preprocessing = get_model(model_name)
        self.model.eval()
        self.head = torch.tensor([])

    def predict(self, x):
        self.embeddings = self.model.encode_image(x) #store latest embeddings computed
        distances = torch.cdist(self.embeddings, torch.cat([c.cur_avg for c in self.centroids.values()], dim = 0))
        prediction = torch.mode(torch.argmin(distances, dim = 1)).values.item()
        return prediction, distances

    def store_new_class(self, label):
        rt = RunningTensor()
        _ = rt.update(self.embeddings.mean(dim = 0).unsqueeze(0))
        self.centroids[label] = rt
        #self.head = nn.parameter.Parameter(torch.cat((self.head, rt.cur_avg.unsqueeze(0)), dim = 0))
    
    def update_class(self, label):
        self.centroids[label].update(self.embeddings.mean(dim = 0).unsqueeze(0))
        #self.head[label] = self.centroids[label].cur_avg
    
    '''def push(self, label):
        if label in self.centroids.keys():
            self._update_class(label)
        else:
            self._store_new_class(label)
    '''

