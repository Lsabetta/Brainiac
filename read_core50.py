import glob
from opt import OPT
from torch.utils.data import Dataset
from PIL import Image

class Core50Dataset(Dataset):
    """ Scenario Dataset for Core50 it requires a scenario number  """
    
    def __init__(self, scenario_id, object_id, data_path='/home/leonardolabs/data', transform=None):
        self.data_path = data_path+'/core50_128x128/'
        self.transform = transform
        self.scenario_id = scenario_id
        self.object_id = object_id
        self.label = self.object_id // 5 + 1
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



if __name__ == '__main__':
    dset = Core50Dataset( 1, 1)
    print('hi')