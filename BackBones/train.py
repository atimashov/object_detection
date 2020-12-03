
import torch
from torch import optim
from torchvision.models import alexnet, vgg16
from dataset import ImageNet
from models import Darknet20
from utils import (
    init_weights,
	get_n_classes,
	get_accuracy,
    # mean_average_precision,
    # intersection_over_union,
    # cellboxes_to_boxes,
    # get_bboxes,
    # plot_image,
    save_checkpoint,
    load_checkpoint,
)
import os
from torch.utils.data import DataLoader
import numpy as np
from time import time, sleep
from datetime import datetime
from termcolor import colored
# from helper import get_params, update_db
import yaml
from tqdm import tqdm
from argparse import  ArgumentParser

seed = 123
torch.manual_seed(seed)

def train_epoch(train_loader, model, optimizer, device, loss_func = torch.nn.CrossEntropyLoss()):
	print(train_loader)
	loop = tqdm(train_loader, leave = True)
	mean_loss, top5_accuracy = [], []

	for batch_idx, (imgs, labels) in enumerate(loop):
		model.train()  # put model to training mode TODO: should I do it every step?
		imgs, labels = imgs.to(device), labels.to(device) # TODO: add dtype
		scores = model(imgs)
		loss = loss_func(scores, labels)

		# Zero out all of the gradients for the variables which the optimizer will update.
		optimizer.zero_grad()

		# Backwards pass and computing gradients
		loss.backward()
		optimizer.step()

		mean_loss.append(loss.item())

		# update progress bar
		loop.set_postfix(loss=loss.item())

	print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

def train_loop(loader, model, epochs = 3, params = None, device = None, loss_func = torch.nn.CrossEntropyLoss, n_tops = [1, 5]):
	L_RATE, DECAY_RATE, DECAY_EPOCHS, WEIGHT_DECAY = params
	optimizer = optim.Adam(model.parameters(), lr = L_RATE, weight_decay = WEIGHT_DECAY)

	for epoch in range(epochs):
		t = time()
		if (epoch + 1) % DECAY_EPOCHS == 0:
			L_RATE *= (1 - DECAY_RATE)
			optimizer = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=WEIGHT_DECAY)


		print(colored('-' * 50, 'cyan'))
		print(colored('{} Epoch {}{} {}'.format('-' * 20, ' ' * (2 - len(str(epoch))), epoch, '-' * 20), 'cyan'))
		print(colored('-' * 50, 'cyan'))

		# TODO: here should be predictions
		tt = time()
		val_acc, val_loss = get_accuracy(loader['val'], model, device, dtype, torch.nn.CrossEntropyLoss(), n_tops)
		metrics = 'loss = {}'.format(val_loss)
		for i, n_top in enumerate(n_tops):
			metrics += ' | TOP{} acc. = {}%'.format(n_top, val_acc[i])
		print(metrics)
		tt = int(time() - tt)
		tt_min, tt_sec = str(tt // 60), str(tt % 60)
		print(colored('It took {}{} min. {}{} sec.'.format(' ' * (2 - len(tt_min)), tt_min, ' ' * (2 - len(tt_sec)), tt_sec), 'cyan'))

		train_epoch(loader['train'], model, optimizer, device, loss_func)
		t = int(time() - t)
		t_min, t_sec = str(t // 60), str(t % 60)
		print()
		print(colored('It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec), 'cyan'))
		print(colored('-' * 50, 'cyan'))
		print()
	tt = time()
	val_acc, val_loss = get_accuracy(loader['val'], model, device, dtype, torch.nn.CrossEntropyLoss(), n_tops)
	metrics = 'loss = {}'.format(val_loss)
	for i, n_top in enumerate(n_tops):
		metrics += ' | TOP{} acc. = {}%'.format(n_top, val_acc[i])
	print(metrics)
	tt = int(time() - tt)
	tt_min, tt_sec = str(tt // 60), str(tt % 60)
	print(colored('It took {}{} min. {}{} sec.'.format(' ' * (2 - len(tt_min)), tt_min, ' ' * (2 - len(tt_sec)), tt_sec), 'cyan'))



if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--model', type=str, default='my_alexnet', help='model name: my_alexnet, alexnet, my_vgg16, vgg16')
	inputs = parser.parse_args()
	print(inputs)

	# get configs
	with open('config.yaml', 'r') as stream:
		cfg = yaml.safe_load(stream)
	device = torch.device('cuda:0' if cfg['GPU'] and torch.cuda.is_available else 'cpu')
	dtype = torch.float32  # TODO: find out how it affects speed and accuracy
	MODEL = cfg['MODEL']
	LOAD_MODEL = cfg['LOAD_MODEL']
	LOAD_MODEL_FILE = cfg['LOAD_MODEL_FILE']
	ROOT = cfg['DATASET_DIR']
	L_RATE = cfg['LEARNING_RATE']
	DECAY_RATE = cfg['DECAY_RATE']
	DECAY_EPOCHS = cfg['DECAY_EPOCHS']
	WEIGHT_DECAY = cfg['WEIGHT_DECAY']
	EPOCHS = cfg['EPOCHS']
	# set up model
	n_classes = get_n_classes(cfg['MIN_IMAGES'], root = ROOT)
	if MODEL == 'Darknet20':
		model = Darknet20(num_classes = n_classes).to(device)
	if LOAD_MODEL:
		checkpoint = torch.load(LOAD_MODEL_FILE)
		model.load_state_dict(checkpoint)
	else:
		model = init_weights(model)

	# create data loader
	data_train = ImageNet(root = cfg['DATASET_DIR'], train = True, min_class = cfg['MIN_IMAGES']) # TODO: add here transform explicitly
	data_val = ImageNet(root = cfg['DATASET_DIR'], train = False, min_class = cfg['MIN_IMAGES'])  # TODO: add here transform explicitly
	data_loader = {
		'train': DataLoader(
			data_train, batch_size = cfg['BATCH_SIZE'], shuffle = True, num_workers = cfg['NUM_WORKERS'],
			drop_last = True, pin_memory = cfg['PIN_MEMORY']
		),
		'val': DataLoader(
			data_val, batch_size = cfg['BATCH_SIZE'], shuffle = True, num_workers = cfg['NUM_WORKERS'],
			drop_last = True, pin_memory = cfg['PIN_MEMORY']
		)
	}

	params = L_RATE, DECAY_RATE, DECAY_EPOCHS, WEIGHT_DECAY
	print(params)
	train_loop(
		loader = data_loader,  model = model, epochs = EPOCHS, params = params, device = device, loss_func = torch.nn.CrossEntropyLoss()
	)
