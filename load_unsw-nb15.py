import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CSVDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        temp = self.data.iloc[index]
        if self.transform:
            temp = self.transform(temp)
        return(temp)
    
unsw_nb15 = CSVDataset(csv_file="F:\pytorch_test\datase_unsw-nb2015\UNSW_NB15_training-set.csv")
dataloader = DataLoader(unsw_nb15,batch_size=64,sampler=True)

print(dataloader)