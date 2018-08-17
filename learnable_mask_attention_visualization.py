import os
import numpy as np
import cv2

mask_dir = "Contrast_0.0001_TV_reg1e-05_mask_LRPatience3_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Aug_16_22_29"

mode = "train"

if mode == "train":
    mode_mask_dir = os.path.join(mask_dir, mode)
elif mode == "test":
    mode_mask_dir = os.path.join(mask_dir, mode)
else:
    raise Exception("no such mode, it shall be either train or test")

video_index = 3

video_frame_length = 22

saved_vis_dir = os.path.join("./mask_visualization",mode_mask_dir)

if not os.path.exists(saved_vis_dir):
    os.makedirs(saved_vis_dir)

video_name=np.load("./saved_weights/"+mask_dir+"/"+mode+"_name.npy")
video_weights = np.load("./saved_weights/"+mask_dir+"/"+mode+"_att_weights.npy")

video_name = video_name.transpose((0,2,1)).reshape(-1,22)

print("video_name.shape: ", video_name.shape)
print("train_weights.shape: ", video_weights.shape)


single_video_name = video_name[video_index]
single_video_weights = video_weights[video_index].reshape(22,7,7)


for img_indx in range(video_frame_length):
    img_path = single_video_name[img_indx]
    print('img_path: ', img_path)
    img_path = img_path.replace("/home/lili/Video","/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d")
    print("img_path:", img_path)
    img = cv2.imread(img_path)
    #height, width, _ = img.shape
    img = cv2.resize(img, (256, 256))

    cam = single_video_weights[img_indx]
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (256, 256))

    heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

    result = heatmap * 0.3 + img* 0.5

    result_name = os.path.join(saved_vis_dir, img_path.split('/')[-3] + img_path.split('/')[-2] + img_path.split('/')[-1])
    print("result_name: ", result_name)
    cv2.imwrite(result_name, result)


