'''
Extract features for hmdb51 dataset

Author: Lili Meng
Date: August 28th, 2018
'''
import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
import os

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader.spatial_dataloader
from utils import *
from network import *
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='hmdb51 spatial stream on resnet101')
parser.add_argument('--num_frames', default=50, type=int, metavar='N', help='number of classes in the dataset')


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        
        return x

def load_frame_per_video(line, video_root_path, num_frames=50):

    video_name = line.split(' ')[0]
    label = int(line.split(' ')[1])
    total_num_imgs_per_video = int(line.split(' ')[2])

    video_path = video_root_path + video_name
  
    img_interval = int(total_num_imgs_per_video/num_frames)
    if img_interval !=0:
        img_index_list = list(range(1, total_num_imgs_per_video+1, img_interval))
    else:
        img_index_list = list(range(1, total_num_imgs_per_video+1))
    
    if len(img_index_list) > num_frames:
        img_index_list = img_index_list[0:num_frames]

    #assert(len(img_index_list)==num_frames)

    imgs_per_video_list = []
    imgs_names_per_video_list = []
    label_per_video_list = []
    print("total_num_imgs_per_video: ", total_num_imgs_per_video)
    for i in range(0, num_frames):

        if total_num_imgs_per_video < num_frames:
            if i >= len(img_index_list):
                img_name = os.path.join(video_path, 'image_' + str('%05d'%(img_index_list[-1])) + '.jpg')
            else:
                img_name = os.path.join(video_path, 'image_' + str('%05d'%(img_index_list[i])) + '.jpg')
        else:
            img_name = os.path.join(video_path, 'image_' + str('%05d'%(img_index_list[i])) + '.jpg')
       

        img = Image.open(img_name).convert('RGB')
        try:
            imgs_per_video_list.append(img)
            imgs_names_per_video_list.append(img_name) 
        except:
            print(os.path.join(path, 'image_' + str('%05d'%(index)) + '.jpg'))
            img.close()

    return imgs_per_video_list, label, imgs_names_per_video_list

def load_frames(img_list, video_root_path, num_frames=50):

    lines = [line.strip() for line in open(img_list).readlines()]

    frames_for_all_video_list = []
    labels_for_all_video_list = []
    frames_names_for_all_video_list = []

    for line in lines:

        video_name = line.split(' ')[0]
        label = int(line.split(' ')[1])
        total_num_imgs_per_video = int(line.split(' ')[2])

        video_path = video_root_path + video_name
      
        img_interval = int(total_num_imgs_per_video/num_frames)
        if img_interval !=0:
            img_index_list = list(range(1, total_num_imgs_per_video+1, img_interval))
        else:
            img_index_list = list(range(1, total_num_imgs_per_video+1))
        
        if len(img_index_list) > num_frames:
            img_index_list = img_index_list[0:num_frames]

        #assert(len(img_index_list)==num_frames)

        imgs_per_video_list = []
        imgs_names_per_video_list = []
        label_per_video_list = []
        for i in range(0, num_frames):

            print("total_num_imgs_per_video: ", total_num_imgs_per_video)
            if total_num_imgs_per_video < num_frames:
                if i >= len(img_index_list):
                    img_name = os.path.join(video_path, 'image_' + str('%05d'%(img_index_list[-1])) + '.jpg')
                else:
                    img_name = os.path.join(video_path, 'image_' + str('%05d'%(img_index_list[i])) + '.jpg')
            else:
                img_name = os.path.join(video_path, 'image_' + str('%05d'%(img_index_list[i])) + '.jpg')
            print("img_name: ", img_name)

            img = Image.open(img_name).convert('RGB')
            try:
                imgs_per_video_list.append(img)
                imgs_names_per_video_list.append(img_name) 
            except:
                print(os.path.join(path, 'image_' + str('%05d'%(index)) + '.jpg'))
                img.close()

        frames_for_all_video_list.append(imgs_per_video_list)
        labels_for_all_video_list.append(label)
        frames_names_for_all_video_list.append(imgs_names_per_video_list)

    return frames_for_all_video_list, frames_names_for_all_video_list, labels_for_all_video_list


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(logits, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = 1
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()


    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():

    global arg
    arg = parser.parse_args()
    print(arg)

    # 1. build model
    print ('==> Build model and setup loss and optimizer')
    # build model
    model = resnet50(pretrained=True, channel=3).cuda()

    #Loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1,verbose=True)

   
    model_resume_path ='./record/spatial_resnet50/model_best.pth.tar'
    # 2. load pretrained model
    if os.path.isfile(model_resume_path):
        print("==> loading checkpoint '{}'".format(model_resume_path))
        checkpoint = torch.load(model_resume_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(model_resume_path, checkpoint['epoch'], best_prec1))
    else:
        print("==> no checkpoint found at '{}'".format(model_resume_path))

    # 3. prepare input data including load all imgs and preprocessing, prepare input tensor
    transform = transforms.Compose([
                transforms.Scale([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # all_frames, all_frame_names, all_labels = load_frames(img_list = "./ucf101_list/new_ucf101_train_list.txt",
 #                                                video_root_path = "/media/dataDisk/THUMOS14/UCF101_jpegs_256/",
 #                                                num_frames=50)

    feature_dir = "/ssd/Lili/hmdb51/saved_features/hmdb51_test"
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    all_logits_list = []
    all_features_list = []

    correct = 0
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    img_list = "./hmdb51_list/new_test.list"

    lines = [line.strip() for line in open(img_list).readlines()]

    i =0
    toal_num_video = len(lines)
    for line in lines:

        imgs_per_video_list, label, imgs_names_per_video_list = load_frame_per_video(line, "/ssd/Lili/hmdb51/", 50)
       
        input_data = torch.stack([transform(frame) for frame in imgs_per_video_list])

        input_var = Variable(input_data.view(-1, 3, input_data.size(2), input_data.size(3)), volatile=True).cuda()

        # 4. extract featrues before the fully connected layer
        features_before_fc = FeatureExtractor(model)

        logits = model(input_var)

        features = features_before_fc(input_var)

        features = features.view(arg.num_frames, 2048, 49)

        logits_np = logits.data.cpu().numpy()

        features_np = np.squeeze(features.data.cpu().numpy())

        
        print("features_np.shape: ", features_np.shape)

        per_video_logits = np.expand_dims(np.mean(logits_np,axis=0), axis=0)
        per_video_label = np.expand_dims(label, axis=0)

        per_video_logits = torch.from_numpy(per_video_logits)
        per_video_label  = torch.from_numpy(per_video_label)

        prec1, prec5 = accuracy(per_video_logits, per_video_label, topk=(1, 5))

        top1.update(prec1[0], 1)
        top5.update(prec5[0], 1)

        print('video {} done, total {}/{}, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
            toal_num_video,top1.avg, top5.avg))

        
        np.save(os.path.join(feature_dir, 'features_{}.npy'.format('%05d'%i)), features_np)
        np.save(os.path.join(feature_dir, 'name_{}.npy'.format('%05d'%i)), imgs_names_per_video_list)
        np.save(os.path.join(feature_dir, 'label_{}.npy'.format('%05d'%i)), per_video_label)

        i+=1
 
if __name__=='__main__':
    main()

