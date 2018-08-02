import os
import numpy as np

def get_train_test_split_videos():
	all_splits_dir = "/home/lili/Video/datasets/hmdb51_org/testTrainMulti_7030_splits"
	train_list_file = "./hmdb51_list/all_train_file.txt"
	test_list_file = "./hmdb51_list/all_test_file.txt"

	f_train = open(train_list_file,'a')
	f_test = open(test_list_file, 'a')

	for file in sorted(os.listdir(all_splits_dir)):
	    if file.endswith(".txt"):
	    	if file.endswith('_test_split1.txt'):
	    		category_name = file.replace('_test_split1.txt', '/')
	    	if file.endswith('_test_split2.txt'):
	    		category_name = file.replace('_test_split2.txt', '/')
	    	if file.endswith('_test_split3.txt'):
	    		category_name = file.replace('_test_split3.txt', '/')
	    	
	    	print("category_name: ", category_name)
	    	abs_file_path = os.path.join(all_splits_dir, file)
	    	lines = [line.strip() for line in open(abs_file_path).readlines()]

	    	for line in lines:
	            video_name = line.split(' ')[0].split('.')[0]
	            train_test = line.split(' ')[1]
	            if  train_test == "1":
	            	f_train.write(category_name + video_name+'\n')
	            if train_test == "2":
	            	f_test.write(category_name + video_name+'\n')
	           

	f_train.close()
	f_test.close()


def get_frame_numbers(video_list_file):

	video_root_dir = "/home/lili/Video/datasets/hmdb51_org/hmdb51_org_frames/"
	
	category_dict = {}
	category_label = 0
	for folder in sorted(os.listdir(video_root_dir)):
		category_dict[folder] = category_label
		category_label+=1

	list_file = "./hmdb51_list/all_new_train_file.txt"
	f_list = open(list_file,'a')
	lines = [line.strip() for line in open(video_list_file).readlines()]

	for line in lines:
		abs_video_dir = os.path.join(video_root_dir, line)
		label = category_dict[line.split('/')[0]]
		f_list.write(line +' '+str(label)+' '+str(len(os.listdir(abs_video_dir)))+'\n')
		
	f_list.close()

def main():

	train_list_file = "./hmdb51_list/all_train_file.txt"
	test_list_file = "./hmdb51_list/all_test_file.txt"

	get_frame_numbers(test_list_file)
	

main()