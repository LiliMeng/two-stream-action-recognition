"""
LSTM model for action recognition
Author: Lili Meng menglili@cs.ubc.ca, March 12th, 2018
"""

from __future__ import print_function
import sys
import os
import math
import shutil
import random
import tempfile
import unittest
import traceback
import torch
import torch.utils.data
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

import argparse

import numpy as np
import time
from PIL import Image
from network import *
use_cuda = True

class Action_Att_LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, seq_len):
		super(Action_Att_LSTM, self).__init__()
		#attention
		self.att_vw = nn.Linear(49*2048, 49, bias=False)
		self.att_hw = nn.Linear(hidden_size, 49)
		self.att_bias = nn.Parameter(torch.zeros(49))
		self.att_w = nn.Linear(49, 1, bias=False)
	
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(input_size, hidden_size)
		self.fc = nn.Linear(hidden_size, output_size)
		self.fc_attention = nn.Linear(hidden_size, seq_len)
		self.fc_out = nn.Linear(hidden_size, output_size)
		self.fc_c0_0 = nn.Linear(2048, 1024)
		self.fc_c0_1 = nn.Linear(1024, 512)
		self.fc_h0_0 = nn.Linear(2048, 1024)
		self.fc_h0_1 = nn.Linear(1024, 512)
		self.input_size = input_size

		self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
		
	def _attention_layer(self, features, hiddens, batch_size):
	  	"""
	  	: param features: (batch_size, 49 *2048)
	  	: param hiddens: (batch_size, hidden_dim)
	  	:return:
	  	"""
	  	#print("features.shape: ", features.shape)
	  	features_tmp = features.contiguous().view(batch_size, -1)
	  	#print("features_tmp.shape: ", features_tmp.shape)
	  	
	  	att_fea = self.att_vw(features_tmp)
	  	#print("att_fea.shape: ", att_fea.shape)
	  	# N-L-D
	  	att_h = self.att_hw(hiddens)
	  	# N-1-D
	  	#att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
	  	#att_out = self.att_w(att_full).squeeze(2)
	  	att_out = att_fea + att_h
	
	  	alpha = nn.Softmax()(att_out)

	  	# N-L
	  	context = torch.sum(features * alpha.unsqueeze(2), 1)
	  
	  	return context, alpha
	
	def get_start_states(self, input_x):

		h0 = torch.mean(torch.mean(input_x,2),2)
		h0 = self.fc_h0_0(h0)
		h0 = self.fc_h0_1(h0)

		c0 = torch.mean(torch.mean(input_x,2),2)
		c0 = self.fc_c0_0(c0)
		c0 = self.fc_c0_1(c0)
		
		return h0, c0
	
	def forward(self, input_x):

		batch_size = input_x.shape[0]

		h0, c0 = self.get_start_states(input_x)

		output_list= []
		for step in range(22):
		
			tmp = input_x[:,:,step,:].transpose(1,2)

			feas, alpha = self._attention_layer(tmp, h0, batch_size)
			h0, c0 = self.lstm_cell(feas, (h0, c0))
			output = self.fc_out(h0) 
			output_list.append(output)

		final_output =  torch.mean(torch.stack(output_list, dim=0),0)
		
		return output

	def init_hidden(self, batch_size):
		result = Variable(torch.zeros(1, batch_size, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result


def lr_scheduler(optimizer, epoch_num, init_lr = 0.001, lr_decay_epochs=10):
	"""Decay learning rate by a factor of 0.1 every lr_decay_epochs.
	"""
	using_cyclic_lr = False
	if using_cyclic_lr == True:
		eta_min = 5e-8
		eta_max = 5e-5

		lr =  eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(epoch_num/FLAGS.max_epoch * np.pi))
	else:
		lr = init_lr *(0.1**(epoch_num//lr_decay_epochs))
		if epoch_num % lr_decay_epochs == 0:
			print("Learning rate changed to be : {}".format(lr))


	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer


def train(batch_size,
		  train_data,
		  train_label,
		  model,
		  model_optimizer,
		  criterion):
	"""
	a training sample which goes through a single step of training.
	"""
	loss = 0
	model_optimizer.zero_grad()

	model_input = (train_data).view(batch_size, -1, FLAGS.num_segments, 49)
	
	model_input = model_input.cuda()
	logits = model.forward(model_input)

	loss += criterion(logits, train_label) 

	loss.backward()

	model_optimizer.step()

	final_loss = loss.data[0]/batch_size

	corrects = (torch.max(logits, 1)[1].view(train_label.size()).data == train_label.data).sum()

	train_accuracy = 100.0 * corrects/batch_size

	return final_loss, train_accuracy

def test_step(batch_size,
			 batch_x,
			 batch_y,
			 model):
	
	test_data_batch = batch_x.view(batch_size, -1, FLAGS.num_segments, 49).cuda()

	test_logits = model(test_data_batch)
	
	corrects = (torch.max(test_logits, 1)[1].view(batch_y.size()).data == batch_y.data).sum()

	test_accuracy = 100.0 * corrects/batch_size

	return test_logits, test_accuracy


def main():

	torch.manual_seed(1234)
	dataset_name = FLAGS.dataset

	maxEpoch = FLAGS.max_epoch

	num_segments = FLAGS.num_segments

	train_data = np.load("./saved_features/spa_train_hmdb51_features.npy")
	train_label = np.load("./saved_features/spa_train_hmdb51_labels.npy")
	train_name = np.load("./saved_features/spa_train_hmdb51_names.npy")

	test_data = np.load("./saved_features/spa_test_hmdb51_features.npy")
	test_label = np.load("./saved_features/spa_test_hmdb51_labels.npy")
	test_name = np.load("./saved_features/spa_test_hmdb51_names.npy")

	print("train_data.shape: ", train_data.shape)

	train_data = torch.from_numpy(train_data)
	train_data = train_data.transpose(1,2)
	train_label = torch.from_numpy(train_label)

	test_data = torch.from_numpy(test_data)
	test_data = test_data.transpose(1,2)
	test_label = torch.from_numpy(test_label)
	print("train_data.shape: ", train_data.shape)

	transform = transforms.Compose([
	            #transforms.Scale([224, 224]),
	            transforms.ToTensor()])

	# replace_with_random_noise_end = False
	# replace_with_random_noise_middle = False

	# noisy_train = torch.randn(train_data.shape[0], train_data.shape[1], 5)
	# noisy_test = torch.randn(test_data.shape[0], test_data.shape[1], 5)

	# if replace_with_random_noise_end == True:
	
	# 	train_data = torch.cat((train_data[:,:,0:10], noisy_train),2)		
	# 	test_data = torch.cat((test_data[:,:,0:10], noisy_test), 2)
	
	# if	replace_with_random_noise_middle == True:

	# 	train_data1 = torch.cat((train_data[:,:,0:5], noisy_train), 2)
	# 	train_data = torch.cat((train_data1, train_data[:,:,10:15]),2)

	# 	test_data1 = torch.cat((test_data[:,:,0:5], noisy_test), 2)
	# 	test_data = torch.cat((test_data1, test_data[:,:,10:15]),2)

	# print("train_data.shape: ", train_data.shape)
	# print("train_label.shape: ", train_label.shape)
	# replicate_frames_5_times = False


	# if replicate_frames_5_times == True:
	# 	train_data = np.repeat(train_data, 2, axis=2)
	# 	train_label = np.repeat(train_label, 2, axis=0)
	# 	test_data = np.repeat(test_data, 2, axis=2)
	# 	test_label = np.repeat(test_label, 2, axis=0)

	lstm_action = Action_Att_LSTM(input_size=2048, hidden_size=512, output_size=51, seq_len=FLAGS.num_segments).cuda()
	model_optimizer = torch.optim.SGD(lstm_action.parameters(), lr=1e-3) 
	#model_optimizer = torch.optim.Adam(lstm_action.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()  

	best_test_accuracy = 0

	num_step_per_epoch_train = train_data.shape[0]//FLAGS.train_batch_size
	num_step_per_epoch_test = test_data.shape[0]//FLAGS.test_batch_size

	
	log_dir = os.path.join('./new_tensorboard_log', 'SGD_spa_att_hidden512'+time.strftime("_%b_%d_%H_%M", time.localtime()))

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	writer = SummaryWriter(log_dir)

	for epoch_num in range(maxEpoch):

		model_optimizer = lr_scheduler(optimizer = model_optimizer, epoch_num=epoch_num, init_lr = 1e-3, lr_decay_epochs=150)
		
		lstm_action.train()
		avg_train_accuracy = 0
		permutation = torch.randperm(train_data.shape[0])
		for i in range(0, train_data.shape[0], FLAGS.train_batch_size):
			indices = permutation[i:i+FLAGS.train_batch_size]
			train_batch_x, train_batch_y, train_batch_name = train_data[indices], train_label[indices], train_name[indices]
			train_batch_x = Variable(train_batch_x).cuda().float()
			train_batch_y = Variable(train_batch_y).cuda().long()
			#print("train_batch_name[0:5] ", train_batch_name[0:5])

			train_loss, train_accuracy = train(FLAGS.train_batch_size, train_batch_x, train_batch_y, lstm_action, model_optimizer, criterion)
			
			avg_train_accuracy+=train_accuracy
			
		final_train_accuracy = avg_train_accuracy/num_step_per_epoch_train
		print("epoch: "+str(epoch_num)+ " train accuracy: " + str(final_train_accuracy))
		writer.add_scalar('train_accuracy', final_train_accuracy, epoch_num)
   

		save_train_file = FLAGS.dataset  + "_numSegments"+str(FLAGS.num_segments)+"_regFactor_"+str(FLAGS.hp_reg_factor)+"_train_acc.txt"
		with open(save_train_file, "a") as text_file:
				print(f"{str(final_train_accuracy)}", file=text_file)

		avg_test_accuracy = 0
		lstm_action.eval()
		for i in range(0, test_data.shape[0], FLAGS.test_batch_size):
			test_indices = range(test_data.shape[0])[i: i+FLAGS.test_batch_size]
			test_batch_x, test_batch_y, test_batch_name = test_data[test_indices], test_label[test_indices], test_name[test_indices]
			test_batch_x = Variable(test_batch_x).cuda().float()
			test_batch_y = Variable(test_batch_y).cuda().long()
			
			test_logits, test_accuracy = test_step(FLAGS.test_batch_size, test_batch_x, test_batch_y, lstm_action)

			avg_test_accuracy+= test_accuracy

	
		final_test_accuracy = avg_test_accuracy/num_step_per_epoch_test
		print("epoch: "+str(epoch_num)+ " test accuracy: " + str(final_test_accuracy))
		writer.add_scalar('test_accuracy', final_test_accuracy, epoch_num)

		save_test_file = FLAGS.dataset  + "_numSegments"+str(FLAGS.num_segments)+"_regFactor_"+str(FLAGS.hp_reg_factor)+"_test_acc.txt"
		with open(save_test_file, "a") as text_file1:
				print(f"{str(final_test_accuracy)}", file=text_file1)

		if final_test_accuracy > best_test_accuracy:
			best_test_accuracy = final_test_accuracy
		print('\033[91m' + "best test accuracy is: " +str(best_test_accuracy)+ '\033[0m') 

	# export scalar data to JSON for external processing
	#writer.export_scalars_to_json("./saved_logs/all_scalars.json")
	writer.close()
			
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HMDB51',
                        help='dataset: "UCF101", "HMDB51"')
    parser.add_argument('--train_batch_size', type=int, default=28,
                    	help='train_batch_size: [64]')
    parser.add_argument('--test_batch_size', type=int, default=30,
                    	help='test_batch_size: [64]')
    parser.add_argument('--max_epoch', type=int, default=200,
                    	help='max number of training epoch: [60]')
    parser.add_argument('--num_segments', type=int, default=22,
                    	help='num of segments per video: [110]')
    parser.add_argument('--use_changed_lr', dest='use_changed_lr',
    					help='not use change learning rate by default', action='store_true')
    parser.add_argument('--use_regularizer', dest='use_regularizer',
    					help='use regularizer', action='store_false')
    parser.add_argument('--hp_reg_factor', type=float, default=0,
                        help='multiply factor for regularization. [0]')
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        raise Exception('Unknown arguments:' + ', '.join(unparsed))
    print(FLAGS)
main()