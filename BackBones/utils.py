import torch
from torch import nn
from torch.nn.init import kaiming_normal_
import os
import json

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


def get_accuracy(loader, model, device, dtype, loss_func = nn.CrossEntropyLoss(), n_tops = [1, 5]):
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
			# print('scores = ', scores.shape, 'labels = ', labels.shape, labels.min(), labels.max())
			# print(scores.min(), scores.max())
			# print('+++')
			# print(labels)
			# print('+++')
			loss = loss_func(scores, labels)

			losses.append(float(loss))

			# accuracy
			_, preds = scores.sort(dim = 1, descending = True)
			correct = preds == labels.view(-1,1)
			for i, n_top in enumerate(n_tops):
				num_correct[i] += correct[:,:n_top].sum()
				num_samples[i] += preds.size(0)

		acc =  list(map(lambda x, y: round(100 * float(x) / y, 2), num_correct, num_samples))
		loss = sum(losses) / len(losses)
	return acc, loss

def get_optimizer(old_params, model, optimizer = None):
	prev_lr, prev_wd, prev_do, prev_opt, flag = old_params
	if optimizer is None:
		if prev_do != 50:
			for i, layer in enumerate(model.features):
				if type(layer) == torch.nn.Dropout:
					model.features[i] = torch.nn.Dropout(prev_do / 100)
			for i, layer in enumerate(model.classifier):
				if type(layer) == torch.nn.Dropout:
					model.classifier[i] = torch.nn.Dropout(prev_do / 100)
		if prev_opt == 'Adam':
			optimizer = optim.Adam(model.parameters(), lr = prev_lr, weight_decay = prev_wd)
		else:
			optimizer = optim.SGD(model.parameters(), lr = prev_lr, weight_decay = prev_wd, momentum = 0.9, nesterov = True)
	else:
		lr, wd, do, opt, flag = get_params()
		if flag:
			return (lr, wd, do, opt, flag), model, optimizer

		if (lr != prev_lr) or (wd != prev_wd) or (do != prev_do) or (opt != prev_opt):
			if prev_do != do:
				for i, layer in enumerate(model.features):
					if type(layer) == torch.nn.Dropout:
						model.features[i] = torch.nn.Dropout(prev_do / 100)
				for i, layer in enumerate(model.classifier):
					if type(layer) == torch.nn.Dropout:
						model.classifier[i] = torch.nn.Dropout(prev_do / 100)
			prev_lr, prev_wd, prev_do, prev_opt = lr, wd, do, opt

			if prev_opt == 'Adam':
				optimizer = optim.Adam(model.parameters(), lr=prev_lr, weight_decay=prev_wd)
			else:
				optimizer = optim.SGD(model.parameters(), lr=prev_lr, weight_decay=prev_wd, momentum=0.9, nesterov=True)
	return (prev_lr, prev_wd, prev_do, prev_opt, flag), model, optimizer


def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print('Saving checkpoint...')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('Loading checkpoint...')
    model.load_state_dict(checkpoint)