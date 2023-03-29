import numpy as np

class Metrics():
    def __init__(self, n_classes):
        # To compute ood Type 1 and Type 2 error
        self.ood_confusion_matrix = np.zeros((2, 2))
        self.cls_confusion_matrix = np.zeros((n_classes, n_classes))
        self.moving_avg_acc = []

    def update(self, pred, gt, known, known_for_real):
        """
        Updates the classification confusion matrix and ood, stores the accuracy 

        Args:
            pred, - int prediction of the model
            gt, int - ground truth label
            known, bool - model thinks it's a known class
            known_for_real, bool - hooman says it's known for real 
        
        """
        if known_for_real:
            self.cls_confusion_matrix[pred, gt] += 1.
            self.moving_avg_acc.append(int(pred == gt))
        self.ood_confusion_matrix[int(known), int(known_for_real)] += 1.
    
    def accuracy(self):
        if self.cls_confusion_matrix.sum() != 0:
            return self.cls_confusion_matrix.diagonal().sum()/self.cls_confusion_matrix.sum()
        else:
            return -1.
    
    def ood(self):
        if self.ood_confusion_matrix[:, 0].sum() != 0:
            return self.ood_confusion_matrix[0, 0] / self.ood_confusion_matrix[:, 0].sum()
        else:
            return -1.
        
    def type1_ood_error(self):
        if self.ood_confusion_matrix[:, 1].sum() != 0:
            return self.ood_confusion_matrix[0, 1] / self.ood_confusion_matrix[:, 1].sum()
        else:
            return -1.
        
    def moving_avg_accuracy(self, time_window=50):
        if len(self.moving_avg_acc) != 0:
            return np.array(self.moving_avg_acc[-time_window:]).mean()
        else:
            return -1.
    
    def class_accuracy(self):
        """ Normalized diagonal of class confusion matrix"""
        return [self.cls_confusion_matrix[i, i]/self.cls_confusion_matrix[:, i].sum() for i in range(self.cls_confusion_matrix.shape[-1])]
    
