import torch as pt
from torch.utils.data import Dataset


class SFDS(Dataset):

    def __init__(self, file_dir):
        super().__init__()
