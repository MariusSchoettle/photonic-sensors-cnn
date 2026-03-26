import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image


class TTI_Dataset(Dataset):
    """
    Reads in assignment file (.csv created with train_cv_test_split.py) and processes for use in torch DataLoader.
    Creates custom datasets for train, CV, or test data. Performs z-score normalization based on stat-dictionary
    passed in, that should provide "mean_temp", "std_temp", "main_time", and "std_time" from the train set.
    If stats is none (i.e. for train set) calculates stats from the data passed in.
    """

    def __init__(
            self, assignment_csv: Path, transform=None, stats=None, normalize_temp=True
    ):
        self.assignment_file = pd.read_csv(assignment_csv)
        self.transform = transform
        self.normalize_temp = normalize_temp

        if stats is None:
            self.stats = {
                "mean_temp": self.assignment_file["temperature"].mean(),
                "std_temp": self.assignment_file["temperature"].std(),
                "mean_time": self.assignment_file["time"].mean(),
                "std_time": self.assignment_file["time"].std(),
            }
        else:
            self.stats = stats

    def __len__(self):
        return len(self.assignment_file)

    def __getitem__(self, index):
        img_path, temperature, time = self.assignment_file.iloc[index].to_list()
        img = Image.open(Path(img_path))
        if self.transform:
            img = self.transform(img)

        # z-score normalization of labels
        if self.normalize_temp:  # only in regression case, not classification
            temperature = (temperature - self.stats["mean_temp"]) / self.stats[
                "std_temp"
            ]
        time = (time - self.stats["mean_time"]) / self.stats["std_time"]

        return img, torch.tensor(data=[temperature, time], dtype=torch.float32)
