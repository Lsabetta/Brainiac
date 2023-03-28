from torchvision import datasets
from tqdm import tqdm

DATA_PATH = '/home/pelosinf/data/'


_ = datasets.CIFAR100(root=DATA_PATH, train=True, download=True)
_ = datasets.CIFAR10(root=DATA_PATH, train=True, download=True)
_ = datasets.Flowers102(root=DATA_PATH, split='train', download=True)
_ = datasets.INaturalist(root=DATA_PATH, version='2017', download=True)
_ = datasets.LFWPeople(root=DATA_PATH, split='10fold', download=True)
_ = datasets.FGVCAircraft(root=DATA_PATH, train=True, download=True)