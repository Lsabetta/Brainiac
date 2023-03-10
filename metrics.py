import numpy as np

class Metrics():
    def __init__(self):
        self.predictions = []
        self.gt = []

        self.matrix = np.zeros((10,10))
        self.ood_num = 0
        self.ood_den = 0
        self.confusion_num = 0
        self.confusion_den = 0

    def update(self, pred, gt, known, known_for_real):
        n = self.matrix.shape[-1]
        if gt+1 > n:
            temp = np.zeros((n+1, n+1))
            temp[:-1, :-1] = self.matrix
            self.matrix = temp
        if known_for_real:
            self.matrix[pred, gt] += 1
        self.update_ood(known, known_for_real)
    
    def accuracy(self):
        return self.matrix.diagonal().sum()/self.matrix.sum()
    
    def class_accuracy(self):
        return [self.matrix[i, i]/self.matrix[:, i].sum() for i in range(self.matrix.shape[-1])]
    
    def ood(self):
        if self.ood != 0:
            return self.ood_num/self.ood_den
        else:
            return None
    
    def confusion(self):
        if self.confusion_den != 0: 
            return self.confusion_num/self.confusion_den
        else:
            return None

    def update_ood(self, known, known_for_real):

        #Era nel dominio del modello e il modello lo ha capito #true negative
        if known and known_for_real: 
            pass

        #Era nel dominio del modello e il modello non lo ha capito #false negative
        if not known and known_for_real: 
            self.confusion_num += 1

        #Non era nel dominio del modello e il modello non lo ha capito #false positive
        if known and not known_for_real: 
            pass

        #Non era nel dominio del modello e il modello lo ha capito #true positive
        if not known and not known_for_real: 
            self.ood_num += 1
        if not known_for_real:
            self.ood_den += 1  
        else:
            self.confusion_den += 1
        #else 0


