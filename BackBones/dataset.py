import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImageNet(Dataset):
	def __init__(self, root = 'home/alex/datasets/imagenet/', train = True):
		self.root = root
		self.train = train
		classes = os.listdir(root)
		with open('{}/classes.json'.format(root)) as json_file:
			classes_names = json.load(json_file)


		min_class = 750 # min amount of images in each class

		self.imgs = []
		self.labels_idx = []
		self.labels_dict = {}
		for my_class in classes:
			imgs = os.listdir('{}/{}'.format(root, my_class))
			if len(imgs) < min_class: continue

			train_val, n_class = int(0.9 * len(imgs)), len(imgs)
			if train:
				self.imgs.extend([os.path.join(root, my_class, img) for img in os.listdir(os.path.join(root, my_class))[:train_val]])
				self.labels_idx.extend([my_class] * train_val)
			else:
				self.imgs.extend([os.path.join(root, my_class, img) for img in os.listdir(os.path.join(root, my_class))[train_val:n_class]])
				self.labels_idx.extend([my_class] * (n_class - train_val))
			self.labels_dict[my_class] = classes_names[my_class]['classname']


	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, idx):
		# load images and targets
		img_path = self.imgs[idx]
		img = Image.open(img_path).convert("RGB")
		target = self.labels_idx[idx]

		if self.train:
			trans = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomCrop(224),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
			])
		else:
			trans = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
		return trans(img), target