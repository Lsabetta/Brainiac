import torch

DSET_N_CLASSES = {

    'CelebA' : 10177,
    'CORE50' : 10,
    'CIFAR100' : 100,
    'CIFAR10' : 10,
    'Flowers102' : 102

              
              }


class OPT:
    SEED = 0

    MODEL = "openCLIP"
    DEVICE='cuda:0'

    DISTANCE_TYPE = "l2"
    DATA_PATH = '/home/luigi/Work/data/'
    PRINT_EVERY = 200
    DATASET = 'CORE50'
    N_CLASSES = DSET_N_CLASSES[DATASET]
    SHUFFLED_SCENARIOS = "shuffled"#/ordered
    OBJ_PER_CLASS = 5
    SELF_LEARNING = False
    THRESHOLDS = [18]#torch.arange(18.5, 19, 1).numpy()
    PROBABILITIES = [1]#torch.arange(0.1, 0.11, 0.1).numpy()
    FRAMES = [1]#[1, 10, 20, 30]
    EMBEDDING_SIZE = {
        'CLIP' : 512,
        'openCLIP' : 1024,          
    }
