import torch
from torch.utils.data import DataLoader
from utils import check_known, set_seeds
from read_core50 import Core50Dataset, prepare_scenario_obj_pairs
from Brainiac import Brainiac
import os
from opt import OPT
from metrics import Metrics
import pickle as pkl
import pandas as pd
from tqdm import tqdm


def _to_core50_label(obj_idx, obj_per_class):
    return (obj_idx-1)//obj_per_class


def main():
    dir_path = f"results/{OPT.DATASET}_{OPT.DISTANCE_TYPE}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}_{OPT.SHUFFLED_SCENARIOS}_p{int(OPT.UPDATE_PROBABILITY*100)}"
    
    # Set the seed of the experiment
    set_seeds(OPT.SEED)
    
    df = pd.DataFrame(columns=['known', 'known_for_real', 'prediction', 'label', 'accuracy', 'ood', 'type1_ood_error', "moving_avg"])

    brainiac = Brainiac(OPT.MODEL, OPT.DISTANCE_TYPE)

    pairs = prepare_scenario_obj_pairs()
    pbar = tqdm(enumerate(pairs), total=len(pairs))
    m = Metrics(OPT.N_CLASSES)
    for iteration, (scenario_id, object_id) in pbar:
        # if iteration>300:
        #     brainiac.distance_type = "normalized_l2"
        # Reads core50 data given a object and scenario
        core_dset = Core50Dataset(scenario_id=scenario_id, object_id=object_id, transform=brainiac.preprocessing)
        
        # If this is the first iteration, ask user for the class name, compute embeddings and store centroid
        # pass None as prediction and ground truth because the model can't know the class 
        if len(brainiac.centroids) == 0:
            label = _to_core50_label(core_dset.object_id, OPT.OBJ_PER_CLASS)
            for (video, labels) in DataLoader(core_dset, batch_size=OPT.PROCESSING_FRAMES):
                embeddings = brainiac.model.encode_image(video.to(OPT.DEVICE))
                break

            brainiac.store_new_centroid(embeddings, label)
            m.update(pred=None, gt=None, known=False, known_for_real=False)
            continue
        
        #compute distances from the known classes and tries to infer the predicted class (argmin(distances)) 
        # iterates 1 time to extract exactly #processing_frames images  
        for (video, labels) in DataLoader(core_dset, batch_size=OPT.PROCESSING_FRAMES):
            model_prediciton, embeddings, distances = brainiac.predict(video.to(OPT.DEVICE))
            break

        #check if the object is known or not, based on the distance from the closest centroid
        known = check_known(model_prediciton, distances, OPT.THRESHOLD)

        ground_truth_label = _to_core50_label(core_dset.object_id, OPT.OBJ_PER_CLASS)
        known_for_real = ground_truth_label in brainiac.label_2_index
            
        #Store the new class or update the centroids based on the interaction with user
        if not known_for_real:
            brainiac.store_new_centroid(embeddings, ground_truth_label)
        else:
            brainiac.update_centroid(embeddings, ground_truth_label, probability=OPT.UPDATE_PROBABILITY, self_learning=OPT.SELF_LEARNING)

        m.update(model_prediciton, brainiac.label_2_index[ground_truth_label], known, known_for_real)
        
        new_row = {'known':known, 
                    'known_for_real':known_for_real, 
                    'prediction':model_prediciton, 
                    'label':ground_truth_label, 
                    'accuracy':m.accuracy(), 
                    'ood':m.ood(), 
                    'type1_ood_error':m.type1_ood_error(),
                    'moving_avg':m.moving_avg_accuracy()}
        df.loc[len(df)] = new_row

        # Logs metrics to csv and prints the progressbar
        if (iteration) % OPT.PRINT_EVERY == 0:
            os.makedirs(dir_path, exist_ok=True)
            pbar.set_description(f"acc:{m.accuracy():.3f} / ood:{m.ood():.3f} / t1:{m.type1_ood_error():.3f} / mva:{m.moving_avg_accuracy():.3f}")
            df.to_csv(f"{dir_path}/metrics_{OPT.THRESHOLD:.2f}.csv", index=False)
            # This saves the matrix as pickle
            with open(f"{dir_path}/matrix_t{OPT.THRESHOLD:.2f}.pkl", "wb") as f:
                pkl.dump(m, f)
    
    print(m.cls_confusion_matrix)
    print(m.ood_confusion_matrix)


if __name__ == "__main__":
    with torch.no_grad():
        for t in OPT.THRESHOLDS:
            for p in OPT.PROBABILITIES:
                for f in OPT.FRAMES:
                    OPT.THRESHOLD = round(t, 2)
                    OPT.UPDATE_PROBABILITY = round(p, 2)
                    OPT.PROCESSING_FRAMES = f
                    main()
    
