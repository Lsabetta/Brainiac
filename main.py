import torch
from torch.utils.data import DataLoader
from utils import check_known, verdict
from read_core50 import Core50Dataset
from Brainiac import Brainiac
import os
from opt import OPT
import glob
import random
from metrics import Metrics


def main():
    #PATH = "/home/leonardolabs/data/core50_128x128/s1/o"
    CLASSES = ["socket", "phone", "scissors", "bulb", "can"]
    
    if OPT.TEST_MODE:
        OBJECT_IDS = [i for i in range(1, 51)]
        random.shuffle(OBJECT_IDS)

    elif not OPT.TEST_MODE:
        OBJECT_IDS = [1+5*i for i in range(5)]

    brainiac = Brainiac(OPT.MODEL)
    label_to_index = {}
    index_to_label = []
    iteration = 1
    m = Metrics()

    ''' 
     if TEST_MODE:
        paths = [[PATH + str(iteration) + "/" + f for f in sorted(os.listdir(PATH + str(iteration)))] for iteration in range(1,51) ]

        labels = torch.tensor([ [ [[i]*OPT.PROCESSING_FRAMES]*5 for i in range(10)]])
        labels = labels.view(-1, labels.shape[-1])
    '''
        
    while True:
        
        print(iteration)
    
        # Core50 paths for the current object
        
        if not OPT.TEST_MODE:
            if iteration < 6:
                object_id = OBJECT_IDS.pop(0)
            else:
                object_id = input("Insert object to extract [1-50]:\n")
            #frames_paths = [PATH + str(iteration) + "/" + f for f in sorted(os.listdir(PATH + str(iteration)))[:OPT.PROCESSING_FRAMES]]
        else:
            object_id = OBJECT_IDS.pop(0)
            print(f'The current object selected is {object_id}')

        core_dset = Core50Dataset(scenario_id=1, object_id=object_id, transform=brainiac.preprocessing)

        
        #if this is the first iteration, ask user for the class name, compute embeddings and store centroid
        if len(brainiac.centroids) == 0:
            first_iteration(core_dset, brainiac, label_to_index, index_to_label)
            iteration += 1
            continue
        
        #compute distances from the known classes and tries to infer the predicted class (argmin(distances)) 
        for (video, labels) in DataLoader(core_dset, batch_size=OPT.PROCESSING_FRAMES):
            model_prediciton, distances = brainiac.predict(video.to(OPT.DEVICE))
            break

        #check if the object is known or not, based on the distance from the closest centroid
        known = check_known(model_prediciton, distances, OPT.THRESHOLD)


        if not OPT.TEST_MODE:
            if iteration < 6:
                known_for_real = False
                ground_truth_label = CLASSES.pop(0)
            else:
                #interact with the user based on if the class is thought already known or not
                known_for_real, ground_truth_label  = verdict(model_prediciton, known, index_to_label, video)

        elif OPT.TEST_MODE:
            known_for_real = core_dset.label in label_to_index
            ground_truth_label = core_dset.label


        
        #Store the new class or update the centroids based on the interaction with user
        if not known_for_real:
            label_to_index[ground_truth_label] = max(label_to_index.values()) + 1
            index_to_label.append(ground_truth_label)
            brainiac.store_new_class(ground_truth_label)
        else:
            brainiac.update_class(ground_truth_label)

        m.update(model_prediciton, ground_truth_label, known, known_for_real)
        if iteration%10 == 0:
            print(f"Total accuracy: {m.accuracy()}\nAccuracy per class: {m.class_accuracy()}\nOOD: {m.ood()}")
        iteration += 1



def first_iteration(core_dset, brainiac, label_to_index, index_to_label):
    if not OPT.TEST_MODE:
        label = input("First class: what is it?\n")
    else:
        label = core_dset[0][1]
    print(f'The current object selected is {label}')

    for (video, labels) in DataLoader(core_dset, batch_size=OPT.PROCESSING_FRAMES):
        brainiac.embeddings = brainiac.model.encode_image(video.to(OPT.DEVICE))
        break

    brainiac.store_new_class(label)
    label_to_index[label] = 0
    index_to_label.append(label)
   

if __name__ == "__main__":
    with torch.no_grad():
        main()
