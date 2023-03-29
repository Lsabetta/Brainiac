import glob
from opt import OPT
from torch.utils.data import Dataset
from PIL import Image
import random 
import cv2


class AllCore50Dataset(Dataset):
    """ Scenario Dataset for Core50 it requires a scenario number  """
    
    def __init__(self, data_path, finegrane=True, transform=None):
        self.data_path = data_path+'/core50_128x128/'
        self.transform = transform
        self.finegrane = finegrane
        self._set_data_and_labels()

    def _set_data_and_labels(self):
        """ Retrieve all paths and labels and shuffle them"""

        # Retrieve all paths of the specified shenario
        self.paths = glob.glob(self.data_path+'/*/*/*.png')
        random.shuffle(self.paths)
        self.labels = self._extract_labels_from_paths(self.paths)
    
    def _extract_labels_from_paths(self, paths):
        labels = []
        for path in paths:
            # Corrects labels starting from 0 to 49
            if self.finegrane:
                labels.append(int(path.split('/')[-2][1:])-1)
            else:
                tmp = int(path.split('/')[-2][1:])-1
                labels.append(tmp // 5)
        return labels
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        x = Image.open(self.paths[index])

        y = self.labels[index]
        if self.transform:
            x = self.transform(x)

        return x, y


class Core50Dataset(Dataset):
    """ Scenario Dataset for Core50 it requires a scenario number  """
    
    def __init__(self, scenario_id, object_id, data_path=OPT.DATA_PATH, transform=None):
        self.data_path = data_path+'/core50_128x128/'
        self.transform = transform
        self.scenario_id = scenario_id
        self.object_id = object_id
        self.label = ((self.object_id - 1) // 5) + 1
        self._set_data_and_labels()

    def _set_data_and_labels(self):
        """ Retrieve all paths and labels"""
        # Retrieve all paths of the specified shenario
        self.paths = sorted(glob.glob(self.data_path+f'/s{self.scenario_id}/o{self.object_id}/*.png'))
        

    def reset_task_to(self, scenario_id):
        """ Reset the dataset to a new scenario"""
        self.scenario_id = scenario_id
        self._set_data_and_labels(scenario_id)


    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        x = Image.open(self.paths[index])
        if self.transform:
            x = self.transform(x)
        y = self.label

        return x, y
    

def prepare_scenario_obj_pairs():
    
    # Shuffles ids
    object_ids = [i for i in range(1, 51)]
    random.shuffle(object_ids)
    
    # Shuffles scenarios
    scenario_ids = [i for i in range(1, 11)]
    random.shuffle(scenario_ids)

    # Couples an object in a scenario
    pairs = [(s, obj_idx)  for s in scenario_ids for obj_idx in object_ids]
    
    # If it is not shuffled the pairs are: 
    #  (scen_2, obj_4), (scen_2, obj_50) ... (scen_2, obj_3) 
    #  (scen_1, obj_4), (scen_1, obj_50) ... (scen_1, obj_3) 
    #  (scen_7, obj_4), (scen_7, obj_50) ... (scen_7, obj_3)
    # 
    # If it is shuffled then there is no structure at ell:
    #  (scen_3, obj_4), (scen_6, obj_50) ... (scen_6, obj_3) 
    # ... all shuffled...
    if OPT.SHUFFLED_SCENARIOS == "shuffled":
        random.shuffle(pairs)

    return pairs

if __name__ == '__main__':
    dset = Core50Dataset( 1, 1)
    print('hi')