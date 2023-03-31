import torch
from torch.utils.data import DataLoader
from utils import check_known, set_seeds
from Brainiac import Brainiac
import os
from opt import OPT
from metrics import Metrics
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import utils

def main():
    dir_path = f"results/{OPT.DATASET}_{OPT.DISTANCE_TYPE}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}_{OPT.SHUFFLED_SCENARIOS}_p{int(OPT.UPDATE_PROBABILITY*100)}_sl{OPT.SELF_LEARNING}"

    # Set the seed of the experiment
    set_seeds(OPT.SEED)

    
    # Define the brainiac
    brainiac = Brainiac(OPT.MODEL, OPT.DISTANCE_TYPE)
    transform = brainiac.preprocessing 
    #transforms.Compose([transforms.Resize((224, 224), im.BICUBIC),transforms.ToTensor()]) 

    # Get loader
    dataset = utils.get_dataset(OPT.DATASET, transform)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    # Metric object
    pbar = tqdm(enumerate(dataset_loader), total=len(dataset_loader))
    m = Metrics(OPT.N_CLASSES)
    df = pd.DataFrame(columns=['known', 'known_for_real', 'prediction', 'label', 'accuracy', 'ood', 'type1_ood_error', "moving_avg"])

    for i, (image, label) in pbar:
        image = image.to(OPT.DEVICE)
        label = label.item()
        
        # Get the prediction of the brainiac
        if i == 0:
            embeddings = brainiac.model.encode_image(image)
            brainiac.store_new_centroid(embeddings, label)
            m.update(pred=None, gt=None, known=False, known_for_real=False)
        else:

            brainiac_prediciton, embeddings, distances = brainiac.predict(image, i)

            #check if the object is known or not, based on the distance from the closest centroid
            known = check_known(brainiac_prediciton, distances, OPT.THRESHOLD)
            
            known_for_real = label in brainiac.label_2_index
        
            #Store the new class or update the centroids based on the interaction with user
            if not known_for_real:
                brainiac.store_new_centroid(embeddings, label)
            else:
                brainiac.update_centroid(embeddings, label, probability=OPT.UPDATE_PROBABILITY, self_learning=OPT.SELF_LEARNING)

            # Update the metrics
            m.update(brainiac_prediciton, brainiac.label_2_index[label], known, known_for_real)

            # append row to dataframe
            new_row = {
                'known':known, 
                'known_for_real':known_for_real, 
                'prediction':brainiac_prediciton, 
                'label':label, 
                'accuracy':m.accuracy(), 
                'ood':m.ood(), 
                'type1_ood_error':m.type1_ood_error(),
                'moving_avg':m.moving_avg_accuracy()
                }
            df.loc[len(df)] = new_row
            
            # Logs metrics to csv and prints the progressbar
            if i % OPT.PRINT_EVERY == 0:
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
                OPT.THRESHOLD = round(t, 2)
                OPT.UPDATE_PROBABILITY = round(p, 2)
                OPT.PROCESSING_FRAMES = 1
                main()
    
