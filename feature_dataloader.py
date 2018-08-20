from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class HMDB51Dataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.dataset = pd.read_csv(csv_file)
      
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feature_file = os.path.join(self.data_dir, self.dataset['Feature'][idx])
        label_file = os.path.join(self.data_dir, self.dataset['Label'][idx])
        name_file = os.path.join(self.data_dir, self.dataset['Name'][idx])
        
        feature_per_video = np.load(feature_file)
        label_per_video = np.load(label_file)
        name_per_video = np.load(name_file)
    
        sample = {'feature': feature_per_video, 'label': label_per_video}
        
        return sample, list(name_per_video)


def get_loader(data_dir, csv_file, batch_size, mode='train', dataset='hmdb51'):
    """Build and return data loader."""

    
    shuffle = True if mode == 'train' else False

    if dataset == 'hmdb51':
        dataset = HMDB51Dataset(data_dir, csv_file)
    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return data_loader

if __name__ == '__main__':
    data_dir = './spa_features/test'

    csv_file = './spa_features/test_features_list.csv'
    batch_size = 30
    data_loader = get_loader(data_dir, csv_file, batch_size=batch_size, mode='test',
                             dataset='hmdb51')

    category_dict = np.load("./category_dict.npy")
   
    for i, (sample, batch_name) in enumerate(data_loader):
        batch_feature = sample['feature']
        batch_label = sample['label']
        print("len(batch_name): ", len(batch_name))
        print("batch_label.shape: ", batch_label.shape)
        
        print("i: ", i)
        #print(batch_img_names)
        break