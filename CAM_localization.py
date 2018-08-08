# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import os
import torch
from network import *

# input image
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    #net = models.resnet50(pretrained=True)
    finalconv_name = 'layer4'
    model_resume_path ='./record/spatial/model_best.pth.tar'
    # 2. load pretrained model
    net = resnet50(pretrained=True, channel=3, num_classes=51).cuda()
    if os.path.isfile(model_resume_path):
        print("==> loading checkpoint '{}'".format(model_resume_path))
        checkpoint = torch.load(model_resume_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        net.load_state_dict(checkpoint['state_dict'])
        print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
          .format(model_resume_path, checkpoint['epoch'], best_prec1))
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    print("output.shape: ", output.shape)
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

print("net._modules.get(finalconv_name): ", net._modules.get(finalconv_name))

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

print("params[-1].cpu().data.numpy().shape: ",params[-1].cpu().data.numpy().shape)
print("params[-2].cpu().data.numpy().shape: ",params[-2].cpu().data.numpy().shape)
print("weight_softmax.shape: ", weight_softmax.shape)

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    #print("feature_conv.shape: ", feature_conv.shape)
    output_cam = []
    print(len(class_idx), class_idx)
    for idx in class_idx:
        #print("weight_softmax[idx].shape: ", weight_softmax[idx].shape)
        print("feature_conv.shape: ", feature_conv.shape)
        print("feature_conv.reshape((nc, h*w)).shape: ", feature_conv.reshape((nc, h*w)).shape)
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        print("cam.shape: ", cam.shape)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        #print("cam_img.shape: ", cam_img.shape)
        #print("cam_img: ", cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    print("len(output_cam): ", len(output_cam))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale((224,224)),
   transforms.ToTensor(),
   normalize
])

# response = requests.get(IMG_URL)
# img_pil = Image.open(io.BytesIO(response.content))
# img_pil.save('test.jpg')


train_name=np.load("./saved_weights/train_name.npy")
train_name = train_name.reshape(280,22)
jump2290_train_name = train_name[1]
img_indx= 1
for img_indx in range(22):
    img_path = jump2290_train_name[img_indx]

    img_pil = Image.open(img_path)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    # download the imagenet category list
    # classes = {int(key):value for (key, value)
    #           in requests.get(LABELS_URL).json().items()}


    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
   
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], idx[i]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, idx)

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s'%idx[0])
    #img = cv2.imread('test.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(CAMs[2], cv2.COLORMAP_JET)
    #heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img* 0.5
    result_name = './CAM_result/CAM_'+img_path.split('/')[-3] + img_path.split('/')[-2] + img_path.split('/')[-1]
    print("result_name: ", result_name)
    cv2.imwrite(result_name, result)
