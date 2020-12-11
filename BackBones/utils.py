import torch
from torch import nn
from torch.nn.init import kaiming_normal_
import os
import json
from termcolor import colored
from datetime import datetime as dt

def init_weights(model):
	for layer in model.features:
		if type(layer) in [nn.Conv2d, nn.Linear]:
			kaiming_normal_(layer.weight)
	for layer in model.classifier:
		if type(layer) in [nn.Conv2d, nn.Linear]:
			kaiming_normal_(layer.weight)
	return model # TODO: should I return?

def get_n_classes(min_class, root = 'home/alex/datasets/imagenet/'):
	classes = os.listdir(root)
	out = 0
	for my_class in classes:
		if '.' in my_class: continue
		imgs = os.listdir('{}/{}'.format(root, my_class))
		out +=  (len(imgs) >= min_class)
	return out


def get_accuracy(loader, model, device, dtype, loss_func = nn.CrossEntropyLoss(), n_tops = [1, 5], max_images = 32500):
	num_correct = [0] * len(n_tops)
	num_samples = [0] * len(n_tops)
	model.eval()  # set model to evaluation mode
	losses = []
	with torch.no_grad():
		for (imgs, labels) in loader:
			imgs = imgs.to(device = device, dtype = dtype)  # move to device, e.g. GPU
			labels = labels.to(device = device, dtype = torch.long)
			# loss
			scores = model(imgs)
			loss = loss_func(scores, labels)
			losses.append(float(loss))

			# accuracy
			_, preds = scores.sort(dim = 1, descending = True)
			correct = preds == labels.view(-1,1)
			preds.shape[0]
			for i, n_top in enumerate(n_tops):
				num_correct[i] += correct[:,:n_top].sum()
				num_samples[i] += preds.size(0)
			if num_samples[0] >= max_images:
				break
		acc =  list(map(lambda x, y: round(100 * float(x) / y, 2), num_correct, num_samples))
		loss = sum(losses) / len(losses)
	return acc, loss

def save_checkpoint(model, cfg,  epoch, loss):
	N, PATH = cfg['SAVE_MODEL_N'], '{}/{}'.format(cfg['SAVE_MODEL_DIR'], cfg['MODEL'])

	# Delete the worst checkpoint
	if len(os.listdir(PATH)) == N:
		max_loss = 0
		file_del = None
		for filename in os.listdir(PATH):
			f = filename.replace('.pth.tar','').split('_')[1:]
			f = float('{}.{}'.format(*f))
			if f > max_loss:
				max_loss = f
				file_del = filename
		if loss < max_loss and os.path.exists('{}/{}'.format(PATH, file_del)):
			os.remove('{}/{}'.format(PATH, file_del))
	if len(os.listdir(PATH)) < N:
		state = {
			'model': model.state_dict(),
			'cfg': cfg,
			'epoch': epoch
		}
		file_save = 'epoch{}{}_{}_{}.pth.tar'.format(
			'0' * (3 - len(str(epoch))), epoch, *str(loss).split('.')
		)
		torch.save(state, '{}/{}'.format(PATH, file_save))


def load_checkpoint(file_name, model):
	print('Loading checkpoint...')
	checkpoint = torch.load(file_name)
	model.load_state_dict(checkpoint['model'])
	cfg = checkpoint['cfg']
	epoch = checkpoint['epoch']
	return cfg, epoch


def print_report(part, epoch = None, t = None, metrics = None):
	if part == 'start':
		# print(colored('-' * 130, 'cyan'))
		# print(colored('{} Epoch {}{} {}'.format('-' * 60, ' ' * (3 - len(str(epoch))), epoch, '-' * 61), 'cyan'))
		# print(colored('-' * 132, 'cyan'))
		# print('-' * 50)
		# print('{} Epoch {}{} {}'.format('-' * 60, ' ' * (3 - len(str(epoch))), epoch, '-' * 61))
		print('{} Epoch {}{} {}'.format(' ' * 60, ' ' * (3 - len(str(epoch))), epoch, ' ' * 61))
		print(' ' * 132)
	elif part == 'end':
		t_min, t_sec = str(t // 60), str(t % 60)
		txt = 'It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec)
		# print()
		# print(colored(txt, 'cyan'))
		# print(colored('-' * 132, 'cyan'))
		print(txt)
		# print('-' * 132)
		print()
		print(colored('-' * 132, 'cyan'))
		print()
	else: # report statistics
		train_loss, val_loss, train_acc, val_acc, n_tops = metrics
		t_loss, v_loss = round(train_loss, 3), round(val_loss, 3)
		t_loss = '{}{}'.format(t_loss, ' ' * (5 - len(str(t_loss))))
		v_loss = '{}{}'.format(v_loss, ' ' * (5 - len(str(v_loss))))
		t_print = 'TRAIN   : loss = {}'.format(t_loss)
		v_print = 'VALIDATE: loss = {}'.format(v_loss)
		for i, n_top in enumerate(n_tops):
			t_acc, v_acc = train_acc[i], val_acc[i]
			if '.' not in str(t_acc):
				t_acc = '{}.00'.format(t_acc)
			else:
					t_acc = '{}{}'.format(t_acc, '0' * (5 - len(str(t_acc))))
			if '.' not in str(v_acc):
				t_acc = '{}.00'.format(v_acc)
			else:
					v_acc = '{}{}'.format(v_acc, '0' * (5 - len(str(v_acc))))
			t_print += ' | TOP{} acc. = {}%'.format(n_top, t_acc)
			v_print += ' | TOP{} acc. = {}%'.format(n_top, v_acc)
		print(t_print)
		print(v_print)
