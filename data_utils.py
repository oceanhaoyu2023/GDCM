import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


class MyDatasets(Dataset):
    def __init__(self, train_input_dir, label_dir, mask_path):
        self.train_input_dir = train_input_dir
        self.label_dir = label_dir
        self.mask_path = mask_path

        self.train_input_list = []
        self.label_list = []

        listdir(self.train_input_dir, self.train_input_list)
        listdir(self.label_dir, self.label_list)

        self.train_input_list.sort()
        self.label_list.sort()
        self.mask_ground = np.load(self.mask_path)

    def __getitem__(self, index):
        train_input_pair = self.train_input_list[index]
        train_label_pair = self.label_list[index]

        train_input_data = np.load(train_input_pair)
        train_label_data = np.load(train_label_pair)
        
        train_label_data[np.where(self.mask_ground[0] > 100)] = np.nan
        train_input_data[np.where(self.mask_ground > 100)] = np.nan
        
        return train_input_data, train_label_data

    def __len__(self):
        return len(self.label_list)