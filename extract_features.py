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

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        
        return x


def load_frames(img_list, video_root_path, num_frames=15):

    lines = [line.strip() for line in open(img_list).readlines()]

    frames_for_all_video_list = []
    labels_for_all_video_list = []
    frames_names_for_all_video_list = []

    for line in lines:

        video_name = line.split(' ')[0]
        label = int(line.split(' ')[1])
        total_num_imgs_per_video = int(line.split(' ')[2])

        video_path = video_root_path + video_name
      
        if total_num_imgs_per_video < num_frames:
            raise Exception("the total number of frames in this video is less than num_frames")
        
        img_interval = int(total_num_imgs_per_video/num_frames)
        img_index_list = list(range(1, total_num_imgs_per_video, img_interval))
        
        if len(img_index_list) > num_frames:
            img_index_list = img_index_list[0:num_frames]

        assert(len(img_index_list)==num_frames)

        imgs_per_video_list = []
        imgs_names_per_video_list = []
        label_per_video_list = []
        for i in range(0, len(img_index_list)):
            img_name = os.path.join(video_path, 'image_' + str('%05d'%(img_index_list[i])) + '.jpg')
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

    # 1. build model
    print ('==> Build model and setup loss and optimizer')
    #build model
    model = resnet50(pretrained=True, channel=3, num_classes=51).cuda()
    #Loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1,verbose=True)

    model_resume_path ='./record/spatial/model_best.pth.tar'
    # 2. load pretrained model
    if os.path.isfile(model_resume_path):
        print("==> loading checkpoint '{}'".format(model_resume_path))
        checkpoint = torch.load(model_resume_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
          .format(model_resume_path, checkpoint['epoch'], best_prec1))
    else:
        print("==> no checkpoint found at '{}'".format(model_resume_path))

    # 3. prepare input data including load all imgs and preprocessing, prepare input tensor
    transform = transforms.Compose([
                transforms.Scale([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    all_frames, all_frame_names, all_labels = load_frames(
                                                img_list = "./hmdb51_list/2class_frame_train.list",
                                                video_root_path = "/home/lili/Video/datasets/HMDB51_concise",
                                                num_frames=15)

    feature_dir = "./saved_features"

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    all_logits_list = []
    all_features_list = []

    correct = 0
    top1 = AverageMeter()
    top5 = AverageMeter()
    toal_num_video = len(all_frames)
    model.eval()
    for i in range(toal_num_video):

        input_data = torch.stack([transform(frame) for frame in all_frames[i]])

        input_var = Variable(input_data.view(-1, 3, input_data.size(2), input_data.size(3)), volatile=True).cuda()
   
        # 4. extract featrues before the fully connected layer
        features_before_fc = FeatureExtractor(model)

        logits = model(input_var)
        features = features_before_fc(input_var)

        logits_np = logits.data.cpu().numpy()
        features_np = np.squeeze(features.data.cpu().numpy())

        all_logits_list.append(logits_np)
        all_features_list.append(features_np)

        
        per_video_logits = np.expand_dims(np.mean(logits_np,axis=0), axis=0)
        per_video_label = np.expand_dims(all_labels[i], axis=0)

        per_video_logits = torch.from_numpy(per_video_logits)
        per_video_label  = torch.from_numpy(per_video_label)

        prec1, prec5 = accuracy(per_video_logits, per_video_label, topk=(1, 5))
    
        top1.update(prec1[0], 1)
        top5.update(prec5[0], 1)
        
        print('video {} done, total {}/{}, moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
                                                                  toal_num_video,top1.avg, top5.avg))
   
        
    all_logits = np.asarray(all_logits_list)
    all_frame_names = np.asarray(all_frame_names)
    all_labels = np.asarray(all_labels)
    all_features = np.asarray(all_features_list)

    np.save(os.path.join(feature_dir,"train_hmdb51_logits.npy"), all_logits)
    np.save(os.path.join(feature_dir,"train_hmdb51_names.npy"),  all_frame_names)
    np.save(os.path.join(feature_dir,"train_hmdb51_labels.npy"), all_labels)
    np.save(os.path.join(feature_dir,"train_hmdb51_features.npy"), all_features)
    

main()
