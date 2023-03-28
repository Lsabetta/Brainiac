import torch
from torch.utils.data import DataLoader
from utils import check_known, verdict, set_seeds
from read_core50 import AllCore50Dataset
from Brainiac import Brainiac
import os
from opt import OPT
import glob
import random
from metrics import Metrics
import pickle as pkl
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd
from torchvision.transforms import InterpolationMode as im




def get_dataset(dset_name, transform):
    
    if dset_name == 'CIFAR100': # 0.65
        return datasets.CIFAR100(root=OPT.DATA_PATH, train=True, download=True, transform=transform)
    
    elif dset_name == 'CIFAR10': # 0.91
        return datasets.CIFAR10(root=OPT.DATA_PATH, train=True, download=True, transform=transform)
    
    elif dset_name == 'Flowers102': #0.558
        return datasets.Flowers102(root=OPT.DATA_PATH, split='train', download=True, transform=transform)
    
    elif dset_name == 'INaturlist': # pesa troppo
        return datasets.INaturalist(root=OPT.DATA_PATH, version='2017', download=True, transform=transform)
    
    elif dset_name == 'LFWPeople': 
        return datasets.LFWPeople(root=OPT.DATA_PATH, split='10fold', download=True, transform=transform)
    
    elif dset_name == 'FGVCAircraft': 
        return datasets.FGVCAircraft(root=OPT.DATA_PATH, train=True, download=True, transform=transform)
    
    elif dset_name == 'CORE50':
        return AllCore50Dataset(OPT.DATA_PATH, finegrane=False, transform=transform)
    else:
        raise ValueError(f"Dataset {dset_name} not supported")


def main():
    dir_path = f"results/{OPT.DATASET}_{OPT.DISTANCE_TYPE}_{OPT.PROCESSING_FRAMES}_{OPT.MODEL}"

    # Set the seed of the experiment
    set_seeds(OPT.SEED)
    
    # Define the brainiac
    brainiac = Brainiac(OPT.MODEL, OPT.DISTANCE_TYPE)
    transform = brainiac.preprocessing 
    #transforms.Compose([transforms.Resize((224, 224), im.BICUBIC),transforms.ToTensor()]) 

    # Get loader
    dataset = get_dataset(OPT.DATASET, transform)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    # Metric object
    pbar = tqdm(enumerate(dataset_loader), total=len(dataset_loader))
    m = Metrics()

    df = pd.DataFrame(columns=['known', 'known_for_real', 'prediction', 'label', 'accuracy', 'ood', 'confusion'])
    for i, (image, label) in pbar:
        image = image.to(OPT.DEVICE)
        label = label.item()
        
        # Get the prediction of the brainiac
        if i == 0:
            brainiac.embeddings = brainiac.model.encode_image(image)
            brainiac.store_new_class(label, image)
            m.update_ood(known=False, known_for_real=False)

        else:

            brainiac_prediciton, distances = brainiac.predict(image)

            #check if the object is known or not, based on the distance from the closest centroid
            known = check_known(brainiac_prediciton, distances, OPT.THRESHOLD)
            
            known_for_real = label in brainiac.label_2_index
        
            #Store the new class or update the centroids based on the interaction with user
            if not known_for_real:
                brainiac.store_new_class(label, image)
            else:
                brainiac.update_class(label)

            # Update the metrics
            m.update(brainiac_prediciton, brainiac.label_2_index[label], known, known_for_real)

            # append row to dataframe
            

            new_row = {'known':known, 
                            'known_for_real':known_for_real, 
                            'prediction':brainiac_prediciton, 
                            'label':label, 
                            'accuracy':m.accuracy(), 
                            'ood':m.ood(), 
                            'confusion':m.confusion()}
            
            df.loc[len(df)] = new_row
            

            if i % OPT.PRINT_EVERY == 0:
                os.makedirs(dir_path, exist_ok=True)

                pbar.set_description(f"Accuracy: {m.accuracy():.3f}")
                df.to_csv(f"{dir_path}/metrics_{format(OPT.THRESHOLD, '.2f')}.csv", index=False)
                #print(f"Total accuracy: {m.accuracy()}")#\nAccuracy per class: {m.class_accuracy()}\nOOD: {m.ood()}\nConfusion: {m.confusion()}")

    print(m.matrix)
    
    # THis saves the matrix
    with open(f"{dir_path}/matrix_t{format(OPT.THRESHOLD, '.2f')}.pkl", "wb") as f:
        pkl.dump(m, f)


if __name__ == "__main__":
    with torch.no_grad():
        #THRESHOLDS = torch.arange(1.05, 1.3, 0.1).numpy()
        #for t in THRESHOLDS:
            #OPT.THRESHOLD = round(t, 2)
        
        OPT.THRESHOLD = 6.6
        main()
    
