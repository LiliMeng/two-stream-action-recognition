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
        video_name = name_per_video[0].split('/')[-2]
        
        img_name_list = []
        for i in range(name_per_video.shape[0]): 
            img_name = name_per_video[i].split('/')[-1].split('.')[0].split('_')[1]
            img_name_list.append(int(img_name))
        img_names = torch.from_numpy(np.asarray(img_name_list))
        
        sample = {'feature': feature_per_video, 'label': label_per_video, 'video_name': video_name, 'img_names':img_names}
        
        return sample


def get_loader(data_dir, csv_file, batch_size, mode='train', dataset='hmdb51'):
    """Build and return data loader."""

    
    shuffle = True if mode == 'train' else False

    if dataset == 'hmdb51':
        dataset = HMDB51Dataset(data_dir, csv_file)
    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    return data_loader

def get_video_names(sample, batch_label, batch_size, category_dict):

    batch_video_name = sample['video_name']
    prev_batch_img_names = sample['img_names']
    batch_img_names = []
    for j in range(batch_size):

        feature_label = batch_label[j].numpy()[0]
        category = category_dict.item().get(feature_label)
            
        img_names = prev_batch_img_names.numpy()[j]
            
        per_video_imgs = []
        for k in range(len(img_names)):
            per_img_name = str('%05d'%img_names[k]) + '.jpg'
            per_img_name = category+'/'+str(batch_video_name[j])+'/img_' + per_img_name
            per_video_imgs.append(per_img_name)
               
        batch_img_names.append(per_video_imgs)

    return batch_img_names

if __name__ == '__main__':
    test_data_dir = './features_sepa'

    test_csv_file = './test_features_list.csv'
    batch_size = 16
    data_loader = get_loader(test_data_dir, test_csv_file, batch_size=batch_size, mode='test',
                             dataset='hmdb51')

    category_dict = np.load("./category_dict.npy")
   
    for i, sample in enumerate(data_loader):
        batch_feature = sample['feature']
        batch_label = sample['label']
        batch_img_names = get_video_names(sample, batch_label, batch_size, category_dict)
        
        print(batch_img_names)
        break