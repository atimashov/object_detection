import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImageNet(Dataset):
	def __init__(
			self, root = 'home/alex/datasets/imagenet/', train = True, min_class = 750,
			balanced = False, augmentations = None
				 ):
		self.root = root
		self.train = train
		self.augmentations = augmentations
		classes = os.listdir(root)
		with open('{}/classes.json'.format(root)) as json_file:
			classes_names = json.load(json_file)

		self.imgs = []
		self.labels_idx = []
		self.labels_dict = {}
		i = 0
		for my_class in classes:
			if '.' in my_class: continue
			imgs = os.listdir('{}/{}'.format(root, my_class))
			if len(imgs) < min_class: continue

			if balanced:
				train_val, n_class = int(0.9 * min_class), min_class
			else:
				train_val, n_class = int(0.9 * len(imgs)), len(imgs)
			if train:
				self.imgs.extend([os.path.join(root, my_class, img) for img in os.listdir(os.path.join(root, my_class))[:train_val]])
				self.labels_idx.extend([i] * train_val)
			else:
				self.imgs.extend([os.path.join(root, my_class, img) for img in os.listdir(os.path.join(root, my_class))[train_val:n_class]])
				self.labels_idx.extend([i] * (n_class - train_val))
			self.labels_dict[my_class] = {
				'name': classes_names[my_class]['classname'],
				'code':my_class
			}
			i += 1


	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, idx):
		# load images and targets
		img_path = self.imgs[idx]
		img = Image.open(img_path).convert("RGB")
		target = self.labels_idx[idx]

		if self.augmentations:
			img = self.augmentations(img)

		return img, target
if __name__ == '__main__':
	pass