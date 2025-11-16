import torch as pt
from torch.utils.data import Dataset
import h5py as hp


class SFDS(Dataset):

    def __init__(self, file_dir):
        super().__init__()
        self.file = hp.File(file_dir, "r")
        self.boards = self.file["board"]
        self.evals = self.file["eval"]

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board_state = pt.tensor(self.boards[idx], dtype=pt.float32)
        evaluation = pt.tensor(self.evals[idx] / 2, dtype=pt.float32)
        return board_state, evaluation
