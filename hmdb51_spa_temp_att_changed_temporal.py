'''
spatial temporal attention for thumos14 object localization dataset

Author: Lili Meng
Date: August 29th, 2018

'''
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
from feature_dataloader import *
from convlstm import *
from utils import *
use_cuda = True

class Action_Att_LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, seq_len):
		super(Action_Att_LSTM, self).__init__()
		#attention
		self.att_vw = nn.Linear(49*2048, 49, bias=False)
		self.att_hw = nn.Linear(hidden_size, 49, bias=False)
		self.att_bias = nn.Parameter(torch.zeros(49))
		self.att_vw_bn= nn.BatchNorm1d(1)
		self.att_hw_bn= nn.BatchNorm1d(1)
		self.hidden_size = hidden_size
		self.fc = nn.Linear(hidden_size, output_size)
		self.fc_attention = nn.Linear(hidden_size, 1)
	
		self.att_feature_w = nn.Linear(2048, 1, bias=False)
		self.att_hidden_w = nn.Linear(hidden_size, 1, bias=False)
		self.fc1 = nn.Linear(2048, output_size)
		self.fc_out = nn.Linear(hidden_size, output_size)
		self.fc_c0_0 = nn.Linear(2048, 1024)
		self.fc_c0_1 = nn.Linear(1024, 512)
		self.fc_h0_0 = nn.Linear(2048, 1024)
		self.fc_h0_1 = nn.Linear(1024, 512)

		self.c0_conv = nn.Sequential(
				nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(1024),
				nn.ReLU(),
				nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(512),
				nn.ReLU())

		self.h0_conv = nn.Sequential(
				nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(1024),
				nn.ReLU(),
				nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(512),
				nn.ReLU())

		self.input_size = input_size

		self.mask_conv = nn.Sequential(
				nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(1024),
				nn.ReLU(),
				nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(512),
				nn.ReLU(),
				nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False),
				nn.Sigmoid(), #(bs*22, 1, 7, 7)
			)

		self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

		self.convlstm_cell = ConvLSTMCell(input_size=(7, 7), 
								input_dim = 2048, 
								hidden_dim = 512,
								kernel_size = (3,3),
								bias = True)

		self.dropout_2d = nn.Dropout2d(p=FLAGS.dropout_ratio)

		self.conv_lstm = ConvLSTM(input_size=(7, 7),
                 			input_dim=2048,
                 			hidden_dim=[512],
                 			kernel_size=(3, 3),
                 			num_layers=1,
                 			batch_first=True,
                 			bias=True,
                 			return_all_layers=True)

	def get_start_states(self, input_x):
		
		h0 = torch.mean(input_x, dim=1)

		c0 = torch.mean(input_x, dim=1)

		h0 = self.h0_conv(h0)
		c0 = self.c0_conv(c0)

		return h0, c0
	
	def temporal_attention_layer(self, features, hiddens):
	  	"""
	  	: param features: (batch_size, 2048, 7, 7)
	  	: param hiddens: (batch_size, hidden_dim)
	  	:return:
	  	"""
	
	  	features_tmp = torch.mean(torch.mean(features, dim=3), dim=2) #[30x2048x7x7]
	  	hiddens_tmp = torch.mean(torch.mean(hiddens, dim=3), dim=2) #[30x512x7x7]
	  	att_fea = self.att_feature_w(features_tmp)
	  	#att_fea = self.att_vw_bn(att_fea)
	  	att_h = self.att_hidden_w(hiddens_tmp)
	  	#att_h = self.att_hw_bn(att_h)
	  	att_out = att_fea + att_h 
	  	#att_out = att_h

	  	return att_out

	def forward(self, input_x):

		batch_size = input_x.shape[0]
		seq_len = input_x.shape[2]
		
		input_x = self.dropout_2d(input_x)
		input_x = input_x.transpose(1,2).contiguous()
	
		input_x = input_x.view(-1, 2048, 7, 7)
		
		mask = self.mask_conv(input_x)
		mask = mask.view(-1, FLAGS.num_segments, 1, 7, 7)

		input_x = input_x.view(-1, FLAGS.num_segments, 2048, 7, 7)

		# calculate total variation regularization (anisotropic version)
		# https://www.wikiwand.com/en/Total_variation_denoising
		diff_i = torch.sum(torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]))
		diff_j = torch.sum(torch.abs(mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]))

		tv_loss = FLAGS.tv_reg_factor*(diff_i + diff_j)

		mask_A = (mask > 0.5).type( torch.cuda.FloatTensor )
		mask_B = (mask < 0.5).type( torch.cuda.FloatTensor )
		contrast_loss = -(mask * mask_A).mean(0).sum() * FLAGS.constrast_reg_factor* 0.5 + (mask * mask_B).mean(0).sum() * FLAGS.constrast_reg_factor * 0.5

		mask_input_x_org = mask * input_x
		
		
		h0, c0 = self.get_start_states(mask_input_x_org)
		
		output_list = []
		
		for i in range(FLAGS.num_segments):
			temporal_att_weight_list = []
			for j in range(FLAGS.num_segments):
				mask_input_x_for_att_per_frame = mask_input_x_org[:,j,:,:,:]
				temporal_att_weight = self.temporal_attention_layer(mask_input_x_for_att_per_frame, h0)

				squeezed_temporal_att_weight = temporal_att_weight.squeeze(dim=1)
				temporal_att_weight_list.append(squeezed_temporal_att_weight.cpu().data.numpy())
			
				temporal_att_weight = temporal_att_weight.view(-1, 1, 1,1)
				
		
			temporal_att_weight =Variable(torch.from_numpy(np.asarray(temporal_att_weight_list).squeeze())).transpose(0,1).cuda()
			
			temporal_att_weight = F.softmax(temporal_att_weight, dim=1) #[30, 50]
			weighted_mask_input_all_frame = torch.sum(mask_input_x_org*(temporal_att_weight.view(-1,FLAGS.num_segments,1,1,1)), dim=1)

			h0, c0 = self.convlstm_cell(weighted_mask_input_all_frame, (h0, c0)) 
			output = torch.mean(torch.mean(h0, dim=3), dim=2)
			output = self.fc_out(output)
		
			output_list.append(output)
		
		final_output = torch.mean(torch.stack(output_list, dim=0),0)

		return final_output, temporal_att_weight, mask, tv_loss, contrast_loss

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

	logits, att_weight, mask, tv_loss, contrast_loss = model.forward(train_data)

	loss += criterion(logits, train_label)

	att_reg = F.relu(att_weight[:, :-2] * att_weight[:, 2:] - att_weight[:, 1:-1].pow(2)).sqrt().mean()
	
	if FLAGS.use_regularizer:
		regularization_loss = FLAGS.hp_reg_factor*att_reg 
		loss += regularization_loss
		loss += tv_loss
		loss += contrast_loss

	loss.backward()

	model_optimizer.step()

	final_loss = loss.data[0]

	corrects = (torch.max(logits, 1)[1].view(train_label.size()).data == train_label.data).sum()

	train_accuracy = 100.0 * corrects/batch_size

	train_pred_label = torch.max(logits, 1)[1].view(train_label.size()).cpu().data.numpy()

	return mask, train_pred_label, final_loss, regularization_loss, tv_loss, contrast_loss, train_accuracy, att_weight, corrects

def test_step(batch_size,
			 batch_x,
			 batch_y,
			 model,
			 criterion):
	
	#print("test_data.shape: ", batch_x.shape)
	test_logits, att_weight, mask, tv_loss, contrast_loss = model.forward(batch_x)
	
	att_reg = F.relu(att_weight[:, :-2] * att_weight[:, 2:] - att_weight[:, 1:-1].pow(2)).sqrt().mean()
	
	corrects = (torch.max(test_logits, 1)[1].view(batch_y.size()).data == batch_y.data).sum()

	test_loss = criterion(test_logits, batch_y)

	test_pred_label = torch.max(test_logits, 1)[1].view(batch_y.size()).cpu().data.numpy()
	if FLAGS.use_regularizer:
		test_reg_loss = FLAGS.hp_reg_factor*att_reg 
		test_loss += test_reg_loss
		test_loss += tv_loss
		test_loss += contrast_loss

	test_accuracy = 100.0 * corrects/batch_size

	return mask, test_pred_label, test_loss, test_reg_loss, tv_loss, contrast_loss, test_accuracy, att_weight, corrects


def main():

	torch.manual_seed(1234)
	dataset_name = FLAGS.dataset

	maxEpoch = FLAGS.max_epoch

	num_segments = FLAGS.num_segments

	# load train data

	# load train data
	train_data_dir = '/ssd/Lili/spa_features/train'
	train_csv_file = '/ssd/Lili/spa_features/train_features_list.csv'


	train_data_loader = get_loader(data_dir=train_data_dir, 
							csv_file = train_csv_file, 
							batch_size = FLAGS.train_batch_size,
							mode ='train',
							dataset='hmdb51')
	# load test data
	test_data_dir = '/ssd/Lili/spa_features/test'
	test_csv_file = '/ssd/Lili/spa_features/test_features_list.csv'
	test_data_loader = get_loader(data_dir = test_data_dir, 
    						csv_file = test_csv_file, 
    						batch_size = FLAGS.test_batch_size, 
    						mode='test',
                            dataset='hmdb51')

	category_dict = np.load("./category_dict.npy")

	lstm_action = Action_Att_LSTM(input_size=2048, hidden_size=512, output_size=51, seq_len=FLAGS.num_segments).cuda() 
	model_optimizer = torch.optim.Adam(lstm_action.parameters(), lr=FLAGS.init_lr, weight_decay=FLAGS.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=model_optimizer, mode='min', patience=FLAGS.lr_patience)
	criterion = nn.CrossEntropyLoss()   

	best_test_accuracy = 0
	
	log_name = 'Contrast_{}_TV_reg{}_mask_LRPatience{}_Adam{}_decay{}_dropout_{}_Temporal_ConvLSTM_hidden512_regFactor_{}'.format(str(FLAGS.constrast_reg_factor), str(FLAGS.tv_reg_factor), str(FLAGS.lr_patience), str(FLAGS.init_lr), str(FLAGS.weight_decay), str(FLAGS.dropout_ratio), str(FLAGS.hp_reg_factor))+time.strftime("_%b_%d_%H_%M", time.localtime())
	#log_name = "tmp"
	log_dir = os.path.join('./Conv_hmdb51_tensorboard', log_name)
	#log_dir = "tmp"
	saved_weights_folder = os.path.join('./saved_weights',log_name)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	writer = SummaryWriter(log_dir)

	if not os.path.exists(saved_weights_folder):
		os.makedirs(saved_weights_folder)

	saved_checkpoint_dir = os.path.join('./saved_checkpoints', log_name)
	if not os.path.exists(saved_checkpoint_dir):
		os.makedirs(saved_checkpoint_dir)

	num_step_per_epoch_train = 3570/FLAGS.train_batch_size
	num_step_per_epoch_test = 1530/FLAGS.test_batch_size

	resumed_checkpoint_dir = "./saved_checkpoints/Contrast_0.0001_TV_reg1e-05_mask_LRPatience5_Adam0.0001_decay0.0001_dropout_0.2_Temporal_ConvLSTM_hidden512_regFactor_1_Sep_12_22_05/thumos14_20_checkpoint_12.pth.tar"
	
	use_pretrained_model = False
	if use_pretrained_model == True:
		checkpoint = torch.load(resumed_checkpoint_dir)
		lstm_action.load_state_dict(checkpoint['model'])

		avg_test_accuracy = 0
		lstm_action.eval()
		test_name_list =[]
		test_tmp_att_weights_list = []
		test_pred_label_list = []
		test_gt_label_list = []
		test_spa_att_weights_list = []
		total_test_corrects = 0
		epoch_test_loss = 0
		epoch_test_reg_loss =0
		epoch_test_tv_loss =0 
		epoch_test_contrast_loss = 0
		for i, (test_sample, test_batch_name) in enumerate(test_data_loader):
		
			test_batch_feature = test_sample['feature'].transpose(1,2)
			test_batch_label = test_sample['label']
			
			test_gt_label_list.append(test_batch_label[:, 0].numpy())
			test_batch_feature = Variable(test_batch_feature, volatile=True).cuda().float()
			test_batch_label = Variable(test_batch_label[:,0], volatile=True).cuda().long()
			
			
			test_mask, test_pred_label, test_loss, test_reg_loss, test_tv_loss, test_contrast_loss, test_accuracy, test_temp_att_weights, test_corrects = test_step(FLAGS.test_batch_size, test_batch_feature, test_batch_label, lstm_action, criterion)

			test_batch_name = np.swapaxes(np.asarray(test_batch_name),0,1)
			test_name_list.append(test_batch_name)

			test_spa_att_weights_list.append(test_mask)
			test_tmp_att_weights_list.append(test_temp_att_weights.cpu().data.numpy())
			test_spa_att_weights_np = torch.cat(test_spa_att_weights_list, dim=0)

			test_pred_label_list.append(test_pred_label)
			print("i: {} batch_test_accuracy: {} ".format(i, test_accuracy))
			total_test_corrects += test_corrects 

			avg_test_accuracy+= test_accuracy

			epoch_test_loss += test_loss

			epoch_test_reg_loss += test_reg_loss
			epoch_test_tv_loss += test_tv_loss
			epoch_test_contrast_loss += test_contrast_loss

		avg_test_corrects = total_test_corrects*100/1530

		epoch_test_loss = epoch_test_loss/num_step_per_epoch_test
		epoch_test_reg_loss = epoch_test_reg_loss/num_step_per_epoch_test
		epoch_test_tv_loss = epoch_test_tv_loss/num_step_per_epoch_test
		epoch_test_contrast_loss = epoch_test_contrast_loss/num_step_per_epoch_test
		#test_tmp_att_weights_np = torch.cat(test_tmp_att_weights_list, dim=0)
		#print("test_spa_att_weights_np.shape ", test_spa_att_weights_np.shape)
		np.save(saved_weights_folder+"/test_name.npy", np.asarray(test_name_list))
		np.save(saved_weights_folder+"/test_att_weights.npy", np.asarray(test_tmp_att_weights_list))
		np.save(saved_weights_folder+"/test_pred_label.npy", np.asarray(test_pred_label_list))
		np.save(saved_weights_folder+"/test_gt_label.npy", np.asarray(test_gt_label_list))
		np.save(saved_weights_folder+"/test_att_weights.npy", test_spa_att_weights_np.cpu().data.numpy())
		final_test_accuracy = avg_test_accuracy/num_step_per_epoch_test

		print("final_test_accuracy: ", final_test_accuracy)
	
	
	for epoch_num in range(maxEpoch):

		lstm_action.train()
		avg_train_accuracy = 0

		train_name_list =[]
		train_spa_att_weights_list = []
		train_gt_label_list = []
		train_pred_label_list = []
		total_train_corrects = 0
		epoch_train_loss = 0 
		epoch_train_reg_loss = 0 
		epoch_train_tv_loss = 0
		epoch_train_contrast_loss = 0
		for i, (train_sample,train_batch_name) in enumerate(train_data_loader):
			
			train_batch_feature = train_sample['feature'].transpose(1,2)
			train_batch_label = train_sample['label']
			train_gt_label_list.append(train_batch_label[:, 0].numpy())
			train_batch_feature = Variable(train_batch_feature).cuda().float()
			train_batch_label = Variable(train_batch_label[:,0]).cuda().long()
			
			train_mask, train_pred_label, train_loss, train_reg_loss, train_tv_loss, train_contrast_loss, train_accuracy, train_spa_att_weights, train_corrects = train(FLAGS.train_batch_size, train_batch_feature, train_batch_label, lstm_action, model_optimizer, criterion)
			#print("train_spa_att_weights[0:5] ",train_spa_att_weights[0:5])
			train_batch_name = np.swapaxes(np.asarray(train_batch_name),0,1)
			train_name_list.append(train_batch_name)
			train_spa_att_weights_list.append(train_mask)
			train_pred_label_list.append(train_pred_label)
			avg_train_accuracy+=train_accuracy
			epoch_train_loss += train_loss
			epoch_train_reg_loss += train_reg_loss
			epoch_train_tv_loss += train_tv_loss
			epoch_train_contrast_loss += train_contrast_loss
			print("batch {}, train_acc: {} ".format(i, train_accuracy))
			total_train_corrects+= train_corrects
			

			
		train_spa_att_weights_np = torch.cat(train_spa_att_weights_list, dim=0)
		avg_train_corrects = total_train_corrects *100 /3570
		epoch_train_loss = epoch_train_loss/num_step_per_epoch_train
		epoch_train_reg_loss = epoch_train_reg_loss/num_step_per_epoch_train
		epoch_train_tv_loss = epoch_train_tv_loss/num_step_per_epoch_train
		epoch_train_contrast_loss = epoch_train_contrast_loss/num_step_per_epoch_train
		#print("train_spa_att_weights_np.shape: ",train_spa_att_weights_np.shape)
		np.save(saved_weights_folder+"/train_name_{}.npy".format('%03d'%epoch_num), np.asarray(train_name_list))
		np.save(saved_weights_folder+"/train_att_weights_{}.npy".format('%03d'%epoch_num), train_spa_att_weights_np.cpu().data.numpy())
		np.save(saved_weights_folder+"/train_pred_label_{}.npy".format('%03d'%epoch_num), np.asarray(train_pred_label_list))
		np.save(saved_weights_folder+"/train_gt_label_{}.npy".format('%03d'%epoch_num), np.asarray(train_gt_label_list))

		final_train_accuracy = avg_train_accuracy/num_step_per_epoch_train
		print("epoch: "+str(epoch_num)+ " train accuracy: " + str(final_train_accuracy))
		print("epoch: "+str(epoch_num)+ " train corrects: " + str(avg_train_corrects))
		writer.add_scalar('train_accuracy', final_train_accuracy, epoch_num)
		writer.add_scalar('train_loss', epoch_train_loss, epoch_num)
		writer.add_scalar('train_tv_loss', epoch_train_tv_loss, epoch_num)
		writer.add_scalar('train_reg_loss', epoch_train_reg_loss, epoch_num)
		writer.add_scalar('train_contrast_loss', epoch_train_contrast_loss, epoch_num)
		
		save_train_file = log_name+"_train_acc.txt"
		with open(save_train_file, "a") as text_file:
				print(f"{str(final_train_accuracy)}", file=text_file)

		

		avg_test_accuracy = 0
		lstm_action.eval()
		test_name_list =[]
		test_tmp_att_weights_list = []
		test_pred_label_list = []
		test_gt_label_list = []
		total_test_corrects = 0
		epoch_test_loss = 0
		epoch_test_reg_loss =0
		epoch_test_tv_loss =0 
		epoch_test_contrast_loss = 0
		for i, (test_sample, test_batch_name) in enumerate(test_data_loader):
		
			test_batch_feature = test_sample['feature'].transpose(1,2)
			test_batch_label = test_sample['label']
			test_gt_label_list.append(test_batch_label[:, 0].numpy())
			
			test_batch_feature = Variable(test_batch_feature, volatile=True).cuda().float()
			test_batch_label = Variable(test_batch_label[:,0], volatile=True).cuda().long()
			

			test_mask, test_pred_label, test_loss, test_reg_loss, test_tv_loss, test_contrast_loss, test_accuracy, test_temp_att_weights, test_corrects = test_step(FLAGS.test_batch_size, test_batch_feature, test_batch_label, lstm_action, criterion)
			test_batch_name = np.swapaxes(np.asarray(test_batch_name),0,1)
			test_name_list.append(test_batch_name)
			test_tmp_att_weights_list.append(test_temp_att_weights)
			test_pred_label_list.append(test_pred_label)
			

			print("i: {} batch_test_accuracy: {} ".format(i, test_accuracy))
			total_test_corrects += test_corrects 

			avg_test_accuracy+= test_accuracy

			epoch_test_loss += test_loss

			epoch_test_reg_loss += test_reg_loss
			epoch_test_tv_loss += test_tv_loss
			epoch_test_contrast_loss += test_contrast_loss

		avg_test_corrects = total_test_corrects*100/1530

		epoch_test_loss = epoch_test_loss/num_step_per_epoch_test
		epoch_test_reg_loss = epoch_test_reg_loss/num_step_per_epoch_test
		epoch_test_tv_loss = epoch_test_tv_loss/num_step_per_epoch_test
		epoch_test_contrast_loss = epoch_test_contrast_loss/num_step_per_epoch_test
		test_tmp_att_weights_np = torch.cat(test_tmp_att_weights_list, dim=0)
		#print("test_spa_att_weights_np.shape ", test_spa_att_weights_np.shape)
		np.save(saved_weights_folder+"/test_name_{}.npy".format('%03d'%epoch_num), np.asarray(test_name_list))
		np.save(saved_weights_folder+"/test_att_weights_{}.npy".format('%03d'%epoch_num), test_tmp_att_weights_np.cpu().data.numpy())
		np.save(saved_weights_folder+"/test_pred_label_{}.npy".format('%03d'%epoch_num), np.asarray(test_pred_label_list))
		np.save(saved_weights_folder+"/test_gt_label_{}.npy".format('%03d'%epoch_num), np.asarray(test_gt_label_list))

		final_test_accuracy = avg_test_accuracy/num_step_per_epoch_test
		print("epoch: "+str(epoch_num)+ " test accuracy: " + str(final_test_accuracy))
		print("epoch: "+str(epoch_num)+ " test corrects: " + str(avg_test_corrects))
		writer.add_scalar('test_accuracy', final_test_accuracy, epoch_num)
		writer.add_scalar('test_loss', epoch_test_loss, epoch_num)
		writer.add_scalar('test_reg_loss', epoch_test_reg_loss, epoch_num)
		writer.add_scalar('test_tv_loss', epoch_test_tv_loss, epoch_num)
		writer.add_scalar('test_contrast_loss', epoch_test_contrast_loss, epoch_num)
		
		scheduler.step(epoch_test_loss.data.cpu().numpy()[0])
		writer.add_scalar('learning_rate', model_optimizer.param_groups[0]['lr'])

		save_test_file = log_name+"_test_acc.txt"
		with open(save_test_file, "a") as text_file1:
				print(f"{str(final_test_accuracy)}", file=text_file1)

		is_best = False
		if final_test_accuracy > best_test_accuracy:
			best_test_accuracy = final_test_accuracy
			is_best = True
		print('\033[91m' + "best test accuracy is: " +str(best_test_accuracy)+ '\033[0m') 
		


		save_checkpoint({'epoch': epoch_num,
			'model': lstm_action.state_dict(),}, is_best=False, save_folder=saved_checkpoint_dir, filename='thumos14_20_checkpoint_{}.pth.tar'.format(epoch_num))
	# export scalar data to JSON for external processing
	#writer.export_scalars_to_json("./saved_logs/all_scalars.json")
	writer.close()
			
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hmdb51',
                        help='dataset: "hmdb51"')
    parser.add_argument('--train_batch_size', type=int, default=30,
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
    parser.add_argument('--tv_reg_factor', type=float, default=0.00001,
                        help='multiply factor for total variation regularization. [0.005]')
    parser.add_argument('--constrast_reg_factor', type=float, default=0.0001,
                        help='constrast regularization factor. [1]')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='initial learning rate. [1e-5]')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay. [1e-5]')
    parser.add_argument('--lr_patience', type=int, default=5,
                    	help='reduce learning rate on plateau patience [3]')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='2d dropout raito. [0.3]')
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        raise Exception('Unknown arguments:' + ', '.join(unparsed))
    print(FLAGS)
main()
