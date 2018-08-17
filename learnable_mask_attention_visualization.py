import os
import numpy as np
import cv2

mask_dir = "Contrast_0.0001_TV_reg1e-05_mask_LRPatience3_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Aug_16_22_29"

saved_vis_dir = os.path.join("./mask_visualization",mask_dir)

if not os.path.exists(saved_vis_dir):
    os.makedirs(saved_vis_dir)

train_name=np.load("./saved_weights/"+mask_dir+"/train_name.npy")
train_weights = np.load("./saved_weights/"+mask_dir+"/train_att_weights.npy")

train_name = train_name.transpose((0,2,1)).reshape(-1,22)

print("train_name")
print(train_name.shape)
print("train_weights.shape")
print(train_weights.shape)
#print(train_name[1])
print(train_weights[1].shape)
jump2290_train_name = train_name[0]
jump2290_train_weights = train_weights[0].reshape(22,7,7)

print(jump2290_train_name.shape)
print(jump2290_train_weights.shape)

img_indx= 1
for img_indx in range(22):
    img_path = jump2290_train_name[img_indx]
    print('img_path: ', img_path)
    img_path = img_path.replace("/home/lili/Video","/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d")
    print("img_path:", img_path)
    img = cv2.imread(img_path)
    #height, width, _ = img.shape
    img = cv2.resize(img, (256, 256))

    cam = jump2290_train_weights[img_indx]
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (256, 256))

    heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

    result = heatmap * 0.3 + img* 0.5

    result_name = saved_vis_dir+'/spa_atten_'+img_path.split('/')[-3] + img_path.split('/')[-2] + img_path.split('/')[-1]
    print("result_name: ", result_name)
    cv2.imwrite(result_name, result)


