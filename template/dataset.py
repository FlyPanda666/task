from torch.utils.data import Dataset
from abc import abstractmethod


class TaskDataset(Dataset):
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
