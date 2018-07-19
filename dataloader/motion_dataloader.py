import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from split_train_test_video import *
import os
 
class motion_dataset(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224

    def stackopf(self):
        # name = 'v_'+self.video
        # u = self.root_dir+ 'u/' + name
        # v = self.root_dir+ 'v/'+ name
        
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)


        for j in range(self.in_channel):
            idx = i + j
            #idx = str(idx)
            #frame_idx = 'frame'+ idx.zfill(6)
            video_path = (self.root_dir + self.video)
            #print(self.root_dir, self.video, video_path)
            h_image = os.path.join(video_path, 'flow_x_' + str('%05d'%(idx)) + '.jpg')
            v_image = os.path.join(video_path, 'flow_x_' + str('%05d'%(idx)) + '.jpg')
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':
            self.video, nb_clips = self.keys[idx].split('-')
            self.clips_idx = random.randint(1,int(nb_clips))
        elif self.mode == 'val':
            self.video,self.clips_idx = self.keys[idx].split('-')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        #label = int(label)-1 
        data = self.stackopf()

        if self.mode == 'train':
            sample = (data,label)
        elif self.mode == 'val':
            sample = (self.video,data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample





class Motion_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel,  path, train_ucf_list, test_ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.train_frame_count = {}
        self.test_frame_count = {}
        self.in_channel = in_channel
        self.data_path=path
        self.train_video = {}
        self.test_video = {}
        self.train_ucf_list = train_ucf_list
        self.test_ucf_list = test_ucf_list
        # split the training and testing videos
        # splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        # self.train_video, self.test_video = splitter.split_video()
        
    def load_frame_count(self, split):
        #print '==> Loading frame number of each video'
        # with open('dic/frame_count.pickle','rb') as file:
        #     dic_frame = pickle.load(file)
        # file.close()

        # for line in dic_frame :
        #     videoname = line.split('_',1)[1].split('.',1)[0]
        #     n,g = videoname.split('_',1)
        #     if n == 'HandStandPushups':
        #         videoname = 'HandstandPushups_'+ g
        #     self.frame_count[videoname]=dic_frame[line] 

        if split == 'train':
            split_list = self.train_ucf_list
        else:
            split_list = self.test_ucf_list
        lines = [line.strip() for line in open(split_list).readlines()]

        for line in lines:
            # videoname = line.split('_',1)[1].split('.',1)[0]
            # n,g = videoname.split('_',1)
            # if n == 'HandStandPushups':
            #     videoname = 'HandstandPushups_'+ g
            # videoname example: "HandstandPushups/v_HandstandPushups_g12_c03"
            videoname = line.split(' ')[0]
            label = int(line.split(' ')[1])
            num_imgs = int(line.split(' ')[2])
            video_path = self.data_path + videoname
            if split == 'train':
                self.train_frame_count[videoname] = num_imgs
                self.train_video[videoname] = label
            else:
                self.test_frame_count[videoname] = num_imgs
                self.test_video[videoname] = label

    def run(self):
        self.load_frame_count('train')
        self.load_frame_count('test')
        self.get_training_dic()
        self.val_sample()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video
            
    def val_sample(self):
        self.dic_test_idx = {}
        #print len(self.test_video)
        for video in self.test_video:
            #n,g = video.split('_',1)

            sampling_interval = int((self.test_frame_count[video]-10)/19)
            for index in range(19):
                clip_idx = index*sampling_interval
                key = video + '-' + str(clip_idx+1)
                self.dic_test_idx[key] = self.test_video[video]
             
    def get_training_dic(self):
        self.dic_video_train={}
        for video in self.train_video:
            
            nb_clips = self.train_frame_count[video]-10
            key = video +'-' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video] 
                            
    def train(self):
        training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print('==> Training data :',len(training_set),' videos',training_set[1][0].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

    def val(self):
        validation_set = motion_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print('==> Validation data :',len(validation_set),' frames',validation_set[1][1].size())
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

# if __name__ == '__main__':
#     data_loader =Motion_DataLoader(BATCH_SIZE=1,num_workers=1,in_channel=10,
#                                         path='/home/ubuntu/data/UCF101/tvl1_flow/',
#                                         ucf_list='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/github/UCF_list/',
#                                         ucf_split='01'
#                                         )
#     train_loader,val_loader,test_video = data_loader.run()
    #print train_loader,val_loader