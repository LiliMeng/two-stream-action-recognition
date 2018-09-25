import os
import numpy as np
import cv2

#mask_dir = "Contrast_0.0001_TV_reg1e-05_mask_LRPatience3_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Aug_16_22_29"

#mask_dir = "Contrast_0_TV_reg0_mask_LRPatience3_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Aug_17_11_48"
#mask_dir = "Contrast_0_TV_reg0_mask_LRPatience3_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Aug_18_16_29"
mask_dir ="noMaskContrast_0.0001_TV_reg1e-05_mask_LRPatience3_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_24_10_54"
mode = "test"

if mode == "train":
    mode_mask_dir = os.path.join(mask_dir, mode)
elif mode == "test":
    mode_mask_dir = os.path.join(mask_dir, mode)
else:
    raise Exception("no such mode, it shall be either train or test")

video_index = 1

video_frame_length = 50

saved_vis_dir = os.path.join("./mask_visualization",mode_mask_dir)

if not os.path.exists(saved_vis_dir):
    os.makedirs(saved_vis_dir)

video_name=np.load("./saved_weights/"+mask_dir+"/"+mode+"_name.npy")
video_weights = np.load("./saved_weights/"+mask_dir+"/"+mode+"_spa_att_weights.npy")

temp_weights = np.load("./saved_weights/"+mask_dir+"/"+mode+"_att_weights.npy")
print("temp_weights.shape: ", temp_weights.shape)
video_name = np.concatenate(video_name, axis=0)
temp_weights = np.concatenate(temp_weights, axis=0)

gt_labels = np.load("./saved_weights/"+mask_dir+"/"+mode+"_gt_label.npy")
pred_labels = np.load("./saved_weights/"+mask_dir+"/"+mode+"_pred_label.npy")
gt_labels = np.concatenate(gt_labels, axis=0)
pred_labels = np.concatenate(pred_labels, axis=0)

print("gt_labels.shape: ", gt_labels.shape)
print("pred_labels.shape: ", pred_labels.shape)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (200,220)
fontScale              = 1.5
fontColor              = (255,255,255)
lineType               = 2

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

for video_index in range(video_name.shape[0]):

    #video_index =1350
    if gt_labels[video_index] == pred_labels[video_index]:
        single_video_name = video_name[video_index]
        single_video_weights = video_weights[video_index].reshape(50,7,7)
        single_temp_weights = temp_weights[video_index]
        #print("single_video_weights.shape: ", single_video_weights.shape)
        
        per_video_name = single_video_name[0].split("/")[-3] +single_video_name[0].split("/")[-2]
        if per_video_name=="sit4638":
            print("single_temp_weight: ", single_temp_weights)
            for img_indx in range(video_frame_length):
                img_path = single_video_name[img_indx]
                print('img_path: ', img_path)
               
                img = cv2.imread(img_path)
                height, width, _ = img.shape
                img = cv2.resize(img, (width, height))

                cam = single_video_weights[img_indx]
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                cam_img = cv2.resize(cam_img, (width, height))

                heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

                result = heatmap * 0.6+ img* 0.5

                result_dir = os.path.join(saved_vis_dir, img_path.split('/')[-3] + img_path.split('/')[-2])
                result_name = img_path.split('/')[-1]
                print("result_name: ", result_name)

                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                scaled_weight = (single_temp_weights[img_indx] - min(single_temp_weights))/(max(single_temp_weights)-min(single_temp_weights))
            
                # cv2.putText(result, str(round(scaled_weight,3)), 
                #     bottomLeftCornerOfText, 
                #     font, 
                #     fontScale,
                #     fontColor,
                #     lineType)

                gray_mask = np.ones((height,width,3), np.uint8)
                gray_mask = gray_mask *scaled_weight*255
                
                img = increase_brightness(img, 30)
                cv2.imwrite(os.path.join(result_dir, 'sit4638_org_'+result_name), img)
                cv2.imwrite(os.path.join(result_dir,'sit4638_'+result_name), result)
            
                #img_changed_brightness = increase_brightness(img, int(scaled_weight*150))
                final_tmp_img = gray_mask*0.9+img*0.1
                cv2.imwrite(os.path.join(result_dir,'sit4638_tmp_'+result_name), final_tmp_img)
                print("single_temp_weight[img_indx]", single_temp_weights[img_indx])
      
