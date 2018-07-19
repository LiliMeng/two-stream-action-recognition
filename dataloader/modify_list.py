import os

train_list = '/home/candice/Documents/datasets/HMDB51_list/list1/train.list'
test_list = '/home/candice/Documents/datasets/HMDB51_list/list1/test.list'
new_train_list = '/home/candice/Documents/datasets/HMDB51_list/list1/new_train.list'
new_test_list = '/home/candice/Documents/datasets/HMDB51_list/list1/new_test.list'

video_dir = '/home/candice/Documents/datasets/hmdb51_clean'

train_lines = [line.strip() for line in open(train_list).readlines()]
test_lines = [line.strip() for line in open(test_list).readlines()]
new_train = open(new_train_list, "w")
new_test = open(new_test_list, "w")

def process_lines(video_dir, lines, new_file):
	for line in lines:
		video_name = line.split('\t')[0]
		print line.split('\t')[0]
		label = int(line.split('\t')[1])
		#subdir = os.path.join(video_dir, video_name)
		subdir = video_dir + video_name
		new_file.write("%s %d %d\n" %(video_name, label, len(os.listdir(subdir))))

process_lines(video_dir, train_lines, new_train)
process_lines(video_dir, test_lines, new_test)