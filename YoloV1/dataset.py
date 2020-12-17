import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
            self, dataset_csv, img_dir, label_dir, S = 7, B = 2, C = 20, transform = None
    ):
        self.annotations = pd.read_csv(dataset_csv)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S, self.B, self.C = S, B, C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace('\n', '').split()
                ]
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform: # TODO: if transforms is None, it will not work
            image = self.transform(image)
            # image, boxes = self.transform(image, boxes) # TODO: make normal augmentation for object detection

        # convert to cells
        label_matrix = torch.zeros((self.S, self.S, self.B * 5 + self.C))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self. S, height * self.S

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return  image, label_matrix

def test():
    img_dir = 'data/images'
    label_dir = 'data/labels'
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])
    data = VOCDataset(
        dataset_csv = 'data/100examples.csv', img_dir = img_dir, label_dir = label_dir , transform = transform
    )
    data_loader = DataLoader(
        data, batch_size = 4, shuffle = True, num_workers = 6, drop_last=True, pin_memory = True
    )
    loop = tqdm(data_loader, leave = True)
    for batch_idx, (imgs, labels) in enumerate(loop):
        loop.set_postfix(imgs_shape=imgs.shape, lables_shape = labels.shape)

if __name__=='__main__':
	test()