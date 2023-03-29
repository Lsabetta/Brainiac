import torch

DSET_N_CLASSES = {

    'CelebA' : 10177,
    'Core50' : 50,
    'CIFAR100' : 100,
    'CIFAR10' : 10,
    'Flowers102' : 102

              
              }

class OPT:
    SEED = 0

    MODEL = "openCLIP"
    DEVICE='cuda:0'

    DISTANCE_TYPE = "l2"
    DATA_PATH = '/home/leonardolabs/data/'
    PRINT_EVERY = 200
    DATASET = 'CelebA'
    N_CLASSES = DSET_N_CLASSES[DATASET]
    SHUFFLED_SCENARIOS = "shuffled"#/ordered
    OBJ_PER_CLASS = 5
    SELF_LEARNING = True
    THRESHOLDS = [6.7]#torch.arange(0.25, 0.55, 0.05).numpy()
    PROBABILITIES = [1]#torch.arange(0., 1.1, 0.1).numpy()
    FRAMES = [1]#[1, 10, 20, 30]