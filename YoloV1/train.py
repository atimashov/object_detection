import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import YoloV1
from dataset import VOCDataset
from augmentations import my_transforms
from utils import (
    init_weights,
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    # cellboxes_to_boxes,
    # get_bboxes,
    plot_image,
    get_dataloader,
    get_metrics,
    get_metrics_NEW,
    print_report,
    # save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import yaml
from tqdm import tqdm
from argparse import ArgumentParser
import os
from time import time
import numpy as np

# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms
#
#     def __call__(self, img, bboxes):
#         for t in self.transforms:
#             img, bboxes = t(img), bboxes
#         return img, bboxes
#
#
# transform_t = Compose([
#     transforms.ColorJitter(brightness = 0.5, hue=.05, saturation=.05),
#     # transforms.RandomGrayscale(p = 0.15),
#     transforms.Resize((448, 448)),
#     transforms.ToTensor(),
# ])
# transform_v = Compose([
#     transforms.Resize((448, 448)),
#     transforms.ToTensor(),
# ])
#
# transform = {
#     'train': transform_t,
#     'val': transform_v
# }

def train_epoch(train_loader, model, optimizer, device, loss_func=torch.nn.CrossEntropyLoss()):
    loop = tqdm(train_loader, leave = True)

    for batch_idx, (imgs, labels) in enumerate(loop):
        model.train()  # put model to training mode TODO: should I do it every step?
        imgs, labels = imgs.to(device), labels.to(device)  # TODO: add dtype
        scores = model(imgs)
        loss = loss_func(scores, labels)

        # Zero out all of the gradients for the variables which the optimizer will update.
        optimizer.zero_grad()
        # Backwards pass and computing gradients
        loss.backward()
        optimizer.step()
        # update progress bar
        loop.set_postfix(loss=loss.item())

def train_loop(cfg_path, gpu_n='0'):
    # get configs
    with open(cfg_path, 'r') as stream:
        config = yaml.safe_load(stream)
    device = torch.device('cuda:{}'.format(gpu_n) if config['GPU'] and torch.cuda.is_available else 'cpu')
    dtype = torch.float32  # TODO: find out how it affects speed and accuracy
    MODEL = config['MODEL']
    LOAD_MODEL = config['LOAD_MODEL']
    LOAD_MODEL_FILE = config['LOAD_MODEL_FILE']
    SAVE_MODEL = config['SAVE_MODEL']
    SAVE_MODEL_N = config['SAVE_MODEL_N']
    SAVE_MODEL_DIR = config['SAVE_MODEL_DIR']
    DATASET_DIR = config['DATASET_DIR']
    L_RATE = config['LEARNING_RATE']
    DECAY_RATE = config['DECAY_RATE']
    DECAY_EPOCHS = config['DECAY_EPOCHS']
    WEIGHT_DECAY = config['WEIGHT_DECAY']
    EPOCHS = config['EPOCHS']
    BATCH_SIZE = config['BATCH_SIZE']
    NUM_WORKERS = config['NUM_WORKERS']
    PIN_MEMORY = config['PIN_MEMORY']
    CSV_TRAIN = config['CSV_TRAIN']
    CSV_VAL = config['CSV_VAL']

    # set up model
    S, B, C = 7, 2, 20 # TODO: add it to config

    if LOAD_MODEL:
        # TODO: load backbone
        model, cfg_save, epoch = load_checkpoint(
            LOAD_MODEL_FILE, device = device, S = S, B = B, C = C,
            cfg=config if MODEL == 'VGG16' else None
        )
        # TODO: init weight
    else:
        if MODEL == 'Darknet':
            model = YoloV1(grid_size=S, num_boxes=B, num_classes=C).to(device)
        elif MODEL == 'VGG':
            pass  # add here VGG backbone
        model = init_weights(model)
    start_epoch = 0

    optimizer = optim.Adam(
        model.parameters(), lr=L_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    loader_params = BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DATASET_DIR, CSV_TRAIN, CSV_VAL
    loader = get_dataloader(loader_params, my_transforms)

    # create folder to save models
    if SAVE_MODEL:
        if not os.path.exists('{}/{}'.format(SAVE_MODEL_DIR, MODEL)):
            os.makedirs('{}/{}'.format(SAVE_MODEL_DIR, MODEL))
    losses, accuracies = {'train': [], 'validate': []}, {'train': [], 'validate': []}

    for epoch in range(start_epoch, EPOCHS + start_epoch):
        t = time()
        if (epoch + 1) % DECAY_EPOCHS == 0:
            L_RATE *= (1 - DECAY_RATE)
            optimizer = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=WEIGHT_DECAY)

        # print epoch number
        print_report(part='start', epoch=epoch)
        # train loop
        train_epoch(loader['train'], model, optimizer, device, loss_fn)

        # print metrics
        train_loss, train_maps = get_metrics_NEW(
            loader = loader['train'], model = model, iou_threshold=0.5, threshold=0.4,
            device = device, loss_func = loss_fn, S = S, B = B, C = C
        )

        val_loss, val_maps = get_metrics_NEW(
            loader = loader['val'], model = model, iou_threshold = 0.5, threshold = 0.4,
            device = device, loss_func = loss_fn
        )

        metrics = train_loss, val_loss, train_maps, val_maps
        print_report(part='accuracy', metrics=metrics)
        # collect metrics
        losses['train'].append(train_loss)
        losses['validate'].append(val_loss)
        accuracies['train'].append(np.mean(train_maps))
        accuracies['validate'].append(np.mean(val_maps))

        # save models
        # if SAVE_MODEL:
        #     save_checkpoint(model=model, cfg=cfg, epoch=epoch, loss=round(val_loss, 3))

        # print time
        print_report(part='end', t=int(time() - t))


if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--gpu-number', type=str, default='0', help='GPU number: 0 or 1')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')

    inputs = parser.parse_args()
    print(inputs)
    gpu_n = inputs.gpu_number
    cfg_path = inputs.config
    train_loop(cfg_path, gpu_n = gpu_n)
