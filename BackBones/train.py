import torch
from torch import optim
from torchvision.models import alexnet, vgg16
from torchvision import transforms
from dataset import ImageNet
from models import Darknet, MyAlexNet
from utils import (
    init_weights,
	get_n_classes,
	get_accuracy,
	print_report,
    # mean_average_precision,
    # intersection_over_union,
    # cellboxes_to_boxes,
    # get_bboxes,
    # plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import SVMLoss, MyEntropy
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



def train_epoch(train_loader, model, optimizer, device, loss_func = torch.nn.CrossEntropyLoss()):
	loop = tqdm(train_loader, leave = True)
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
		# update progress bar
		loop.set_postfix(loss=loss.item())

def train_loop(loader, model, epochs = 3, start_epoch = 0, params = None, device = None, loss_func = torch.nn.CrossEntropyLoss, n_tops = [1, 5]):
	L_RATE, DECAY_RATE, DECAY_EPOCHS, WEIGHT_DECAY, SAVE_MODEL, SAVE_MODEL_N, SAVE_MODEL_DIR, MODEL, N_LAYERS = params
	optimizer = optim.Adam(model.parameters(), lr = L_RATE, weight_decay = WEIGHT_DECAY)
	if SAVE_MODEL:
		if MODEL == 'Darknet':
			path = '{}{}'.format(MODEL, N_LAYERS)
		else:
			path = MODEL
		if not os.path.exists('{}/{}'.format(SAVE_MODEL_DIR, path)):
			os.makedirs('{}/{}'.format(SAVE_MODEL_DIR, path))
	losses, accuracies = {'train': [], 'validate': []}, {'train': [], 'validate': []}

	for epoch in range(start_epoch, epochs + start_epoch):
		t = time()
		if (epoch + 1) % DECAY_EPOCHS == 0:
			L_RATE *= (1 - DECAY_RATE)
			optimizer = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=WEIGHT_DECAY)

		# print epoch number
		print_report(part = 'start', epoch = epoch)
		# train loop
		train_epoch(loader['train'], model, optimizer, device, loss_func)
		# print metrics
		val_acc, val_loss = get_accuracy(loader['val'], model, device, dtype, loss_func, n_tops)
		train_acc, train_loss = get_accuracy(loader['train'], model, device, dtype, loss_func, n_tops)
		metrics = train_loss, val_loss, train_acc, val_acc, n_tops
		print_report(part='accuracy', metrics = metrics)
		# collect metrics
		losses['train'].append(train_loss)
		losses['validate'].append(val_loss)
		accuracies['train'].append(train_acc)
		accuracies['validate'].append(val_acc)

		# save models
		if SAVE_MODEL:
			save_checkpoint(model = model, cfg = cfg, epoch = epoch, loss = round(val_loss, 3))
		
		# print time
		print_report(part='end', t = int(time() - t))

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--gpu-number', type=str, default='0', help='GPU number: 0 or 1')
	parser.add_argument('--config', type=str, default='config.yaml', help='config file')

	inputs = parser.parse_args()
	print(inputs)
	gpu_n = inputs.gpu_number
	cfg_path = inputs.config

	seed = 123
	torch.manual_seed(seed)
	# get configs
	with open(cfg_path, 'r') as stream:
		cfg = yaml.safe_load(stream)
	device = torch.device('cuda:{}'.format(gpu_n) if cfg['GPU'] and torch.cuda.is_available else 'cpu')
	dtype = torch.float32  # TODO: find out how it affects speed and accuracy
	MODEL = cfg['MODEL']
	LOAD_MODEL = cfg['LOAD_MODEL']
	LOAD_MODEL_FILE = cfg['LOAD_MODEL_FILE']
	SAVE_MODEL = cfg['SAVE_MODEL']
	SAVE_MODEL_N = cfg['SAVE_MODEL_N']
	SAVE_MODEL_DIR = cfg['SAVE_MODEL_DIR']
	DATASET_DIR = cfg['DATASET_DIR']
	L_RATE = cfg['LEARNING_RATE']
	DECAY_RATE = cfg['DECAY_RATE']
	DECAY_EPOCHS = cfg['DECAY_EPOCHS']
	WEIGHT_DECAY = cfg['WEIGHT_DECAY']
	EPOCHS = cfg['EPOCHS']
	BALANCED = cfg['BALANCED_DATASET']
	BATCH_SIZE = cfg['BATCH_SIZE']
	NUM_WORKERS = cfg['NUM_WORKERS']
	PIN_MEMORY = cfg['PIN_MEMORY']
	MIN_IMAGES = cfg['MIN_IMAGES']
	LOSS = cfg['LOSS']
	N_LAYERS = cfg['N_LAYERS']
	if LOSS =='SVM':
		loss_fc = SVMLoss()
	elif LOSS == 'My Entropy':
		loss_fc = MyEntropy()
	else:
		loss_fc = torch.nn.CrossEntropyLoss()


	# set up model
	n_classes = get_n_classes(MIN_IMAGES, root = DATASET_DIR)
	if MODEL == 'Darknet':
		N_LAYERS = max(20, min(24, N_LAYERS))
		model = Darknet(n_layers = N_LAYERS, num_classes = n_classes).to(device)
	elif MODEL == 'Alexnet':
		model = MyAlexNet(num_classes = n_classes).to(device)
	if LOAD_MODEL:
		cfg_cp, start_epoch = load_checkpoint(LOAD_MODEL_FILE, model)
		val = input('Do you want to use config from checkpoint? Answer "yes" or "no": ')
		if 'val' == 'yes':
			L_RATE = cfg_cp['LEARNING_RATE']
			DECAY_RATE = cfg_cp['DECAY_RATE']
			DECAY_EPOCHS = cfg_cp['DECAY_EPOCHS']
			WEIGHT_DECAY = cfg_cp['WEIGHT_DECAY']
			BALANCED = cfg_cp['BALANCED_DATASET']
			BATCH_SIZE = cfg_cp['BATCH_SIZE']
			NUM_WORKERS = cfg_cp['NUM_WORKERS']
			PIN_MEMORY = cfg_cp['PIN_MEMORY']
			MIN_IMAGES = cfg_cp['MIN_IMAGES']
			LOSS = cfg_cp['LOSS']
	else:
		model = init_weights(model)
		start_epoch = 0

	# create data loader
	my_augment = {
		'train': transforms.Compose([
			# transforms.ToPILImage(), # TODO: do I need it?
			transforms.Resize((256, 256)),
			transforms.RandomCrop(224),
			transforms.ColorJitter(brightness = 0.5, hue=.05, saturation=.05),
			transforms.RandomRotation(degrees = 45),
			transforms.RandomHorizontalFlip(0.5),
			# transforms.RandomVerticalFlip(0.5), # TODO: test on corrosion
			transforms.RandomGrayscale(p = 0.15),
			transforms.ToTensor(), # TODO: Does normalize do it?
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
	}

	data_train = ImageNet(
		root = cfg['DATASET_DIR'], train = True, min_class = cfg['MIN_IMAGES'],
		balanced = BALANCED, augmentations= my_augment['train']
	) # TODO: add here transform explicitly
	data_val = ImageNet(
		root = cfg['DATASET_DIR'], train = False, min_class = cfg['MIN_IMAGES'],
		balanced = BALANCED, augmentations= my_augment['val']
	)  # TODO: add here transform explicitly
	data_loader = {
		'train': DataLoader(
			data_train, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS,
			drop_last = True, pin_memory = PIN_MEMORY
		),
		'val': DataLoader(
			data_val, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS,
			drop_last = True, pin_memory = PIN_MEMORY
		)
	}

	params = L_RATE, DECAY_RATE, DECAY_EPOCHS, WEIGHT_DECAY, SAVE_MODEL, SAVE_MODEL_N, SAVE_MODEL_DIR, MODEL, N_LAYERS
	print()
	train_loop(
		loader = data_loader,  model = model, epochs = EPOCHS, start_epoch = start_epoch, params = params, device = device, loss_func = loss_fc
	)
