import torch
from torch.utils.data import DataLoader
from utils import check_known, verdict, set_seeds
from read_core50 import Core50Dataset
from Brainiac import Brainiac
import os
from opt import OPT
import glob
import random
from metrics import Metrics
import pickle as pkl
import streamlit as st
import pandas as pd
CORE50_CLASSES = ["socket", "phone", "scissors", "bulb", "can", "glasses", "ball", "marker", "mug", "remote"]
CORE_50_CLASSES_DICTIONARY = {i: CORE50_CLASSES[i] for i in range(10)}
def _to_core50_label(obj_idx):
    return CORE_50_CLASSES_DICTIONARY[(obj_idx-1)//5]

def main():
    set_seeds(OPT.SEED)
    
    df = pd.DataFrame(columns=['known', 'known_for_real', 'prediction', 'label', 'accuracy', 'ood', 'confusion'])
   
    OBJECT_IDS = [i for i in range(1, 51)]
    random.shuffle(OBJECT_IDS)
    SCENARIOS_IDS = [i for i in range(1, 11)]
    random.shuffle(SCENARIOS_IDS)
    pairs = [(s, obj_idx)  for s in SCENARIOS_IDS for obj_idx in OBJECT_IDS]
    if OPT.SHUFFLED_SCENARIOS == "shuffled":
        random.shuffle(pairs)
 
    if OPT.WEBAPP:
        brainiac = st.session_state.brainiac
    else:
        brainiac = Brainiac(OPT.MODEL, OPT.DISTANCE_TYPE)

    iteration = 1
    m = Metrics()
    print_every = 1 if OPT.VERBOSE else 100
    for iteration, (scenario_id, object_id) in enumerate(pairs):
        if iteration ==100:
            break
        if (iteration) % print_every == 0:
            print(f"Iteration n {iteration}")
            print(f"Unique classes seen: {len(brainiac.label_2_index)}, threshold: {format(OPT.THRESHOLD, '.2f')}")
            print(f"{brainiac.label_2_index=}\n{brainiac.index_2_label=}")
        
        if OPT.VERBOSE:
            print(f'The current object selected is {object_id}, scenario {scenario_id}')
        core_dset = Core50Dataset(scenario_id=scenario_id, object_id=object_id, transform=brainiac.preprocessing)

            
        #if this is the first iteration, ask user for the class name, compute embeddings and store centroid
        if len(brainiac.centroids) == 0:
            label = _to_core50_label(core_dset.object_id)

            for (video, labels) in DataLoader(core_dset, batch_size=OPT.PROCESSING_FRAMES):
                brainiac.embeddings = brainiac.model.encode_image(video.to(OPT.DEVICE))
                break

            brainiac.store_new_class(label, video[0])
            m.update_ood(known=False, known_for_real=False)
            continue
        
        #compute distances from the known classes and tries to infer the predicted class (argmin(distances)) 
        for (video, labels) in DataLoader(core_dset, batch_size=OPT.PROCESSING_FRAMES):
            model_prediciton, distances = brainiac.predict(video.to(OPT.DEVICE))
            break

        #check if the object is known or not, based on the distance from the closest centroid
        known = check_known(model_prediciton, distances, OPT.THRESHOLD)


   
        ground_truth_label = _to_core50_label(core_dset.object_id)
        known_for_real = ground_truth_label in brainiac.label_2_index
        


        
        #Store the new class or update the centroids based on the interaction with user
        if not known_for_real:
            #brainiac.label_2_index[ground_truth_label] = len(brainiac.label_2_index.keys())
            brainiac.store_new_class(ground_truth_label, video[0] )
        else:
            brainiac.update_class(ground_truth_label)

        m.update(model_prediciton, brainiac.label_2_index[ground_truth_label], known, known_for_real)
        
        new_row = {'known':known, 
                    'known_for_real':known_for_real, 
                    'prediction':model_prediciton, 
                    'label':ground_truth_label, 
                    'accuracy':m.accuracy(), 
                    'ood':m.ood(), 
                    'confusion':m.confusion()}
        
        df.loc[len(df)] = new_row

        if (iteration) % print_every == 0:
            print(f"Total accuracy: {m.accuracy()}\nAccuracy per class: {m.class_accuracy()}\nOOD: {m.ood()}\nConfusion: {m.confusion()}")
        
    print(m.matrix)
    dir_path = f"results/{OPT.DATASET}_{OPT.DISTANCE_TYPE}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}_{OPT.SHUFFLED_SCENARIOS}_p{int(OPT.UPDATE_PROBABILITY*100)}"
    os.makedirs(dir_path, exist_ok=True)
    df.to_csv(f"{dir_path}/metrics_{format(OPT.THRESHOLD, '.2f')}.csv", index=False)
    
    with open(f"{dir_path}/matrix_t{format(OPT.THRESHOLD, '.2f')}.pkl", "wb") as f:
        pkl.dump(m, f)


   

if __name__ == "__main__":
    with torch.no_grad():
        main()
    
