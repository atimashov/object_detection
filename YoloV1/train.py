import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import YoloV1
from dataset import VOCDataset
from utils import (
    init_weights,
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    get_dataloader,
    print_report,
    # save_checkpoint,
    # load_checkpoint,
)
from loss import YoloLoss
import yaml
from tqdm import tqdm
from argparse import ArgumentParser
import os
from time import time

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16  # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


# def train_fn(train_loader, model, optimizer, loss_fn):
#     loop = tqdm(train_loader, leave=True)
#     mean_loss = []
#
#     for batch_idx, (x, y) in enumerate(loop):
#         x, y = x.to(DEVICE), y.to(DEVICE)
#         out = model(x)
#         loss = loss_fn(out, y)
#         mean_loss.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # update progress bar
#         loop.set_postfix(loss=loss.item())
#
#     print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def train_epoch(train_loader, model, optimizer, device, loss_func=torch.nn.CrossEntropyLoss()):
    loop = tqdm(train_loader, leave=True)

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


# def main():
#     model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
#     optimizer = optim.Adam(
#         model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
#     )
#     loss_fn = YoloLoss()
#
#     if LOAD_MODEL:
#         load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
#     else:
#
#
#     train_dataset = VOCDataset(
#         "data/100examples.csv",
#         transform=transform,
#         img_dir=IMG_DIR,
#         label_dir=LABEL_DIR,
#     )
#
#     test_dataset = VOCDataset(
#         "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
#     )
#
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         shuffle=True,
#         drop_last=True,
#     )
#
#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY,
#         shuffle=True,
#         drop_last=True,
#     )
#
#     for epoch in range(EPOCHS):
#         # for x, y in train_loader:
#         #    x = x.to(DEVICE)
#         #    for idx in range(8):
#         #        bboxes = cellboxes_to_boxes(model(x))
#         #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
#         #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
#
#         #    import sys
#         #    sys.exit()
#
#         pred_boxes, target_boxes = get_bboxes(
#             train_loader, model, iou_threshold=0.5, threshold=0.4
#         )
#
#         mean_avg_prec = mean_average_precision(
#             pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
#         )
#         print(f"Train mAP: {mean_avg_prec}")
#
#         #if mean_avg_prec > 0.9:
#         #    checkpoint = {
#         #        "state_dict": model.state_dict(),
#         #        "optimizer": optimizer.state_dict(),
#         #    }
#         #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
#         #    import time
#         #    time.sleep(10)
#
#         train_fn(train_loader, model, optimizer, loss_fn)


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
    if MODEL == 'Darknet':
        model = YoloV1(grid_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    elif MODEL == 'VGG':
        pass  # add here VGG backbone
    if LOAD_MODEL:
        # TODO: load backbone
        # cfg_cp, start_epoch = load_checkpoint(LOAD_MODEL_FILE, model)
        val = input('Do you want to use config from checkpoint? Answer "yes" or "no": ')
        # if 'val' == 'yes':
        #     L_RATE = cfg_cp['LEARNING_RATE']
        #     DECAY_RATE = cfg_cp['DECAY_RATE']
        #     DECAY_EPOCHS = cfg_cp['DECAY_EPOCHS']
        #     WEIGHT_DECAY = cfg_cp['WEIGHT_DECAY']
        #     BALANCED = cfg_cp['BALANCED_DATASET']
        #     BATCH_SIZE = cfg_cp['BATCH_SIZE']
        #     NUM_WORKERS = cfg_cp['NUM_WORKERS']
        #     PIN_MEMORY = cfg_cp['PIN_MEMORY']
        #     MIN_IMAGES = cfg_cp['MIN_IMAGES']
        #     LOSS = cfg_cp['LOSS']
    else:
        model = init_weights(model)
        start_epoch = 0

    optimizer = optim.Adam(
        model.parameters(), lr=L_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    loader_params = BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DATASET_DIR, CSV_TRAIN, CSV_VAL
    loader = get_dataloader(loader_params)

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
        pred_bb, target_bb = get_bboxes(
            loader['train'], model, iou_threshold=0.5, threshold=0.4
        )
        train_map = mean_average_precision(
            pred_bb, target_bb, iou_threshold = 0.5, box_format = 'midpoint'
        )

        v_pred_bb, v_target_bb = get_bboxes(
            loader['val'], model, iou_threshold=0.5, threshold=0.4
        )
        val_map = mean_average_precision(
            v_pred_bb, v_target_bb, iou_threshold=0.5, box_format='midpoint'
        )

        metrics = -1, -1, train_map, val_map
        print_report(part='accuracy', metrics=metrics)
        # collect metrics
        # losses['train'].append(train_loss)
        # losses['validate'].append(val_loss)
        # accuracies['train'].append(train_acc)
        # accuracies['validate'].append(val_acc)

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
    train_loop(cfg_path, gpu_n='0')
# 	with open(cfg_path, 'r') as stream:
# 		cfg = yaml.safe_load(stream)
# 	device = torch.device('cuda:{}'.format(gpu_n) if cfg['GPU'] and torch.cuda.is_available else 'cpu')
# 	dtype = torch.float32  # TODO: find out how it affects speed and accuracy
# 	MODEL = cfg['MODEL']
# 	LOAD_MODEL = cfg['LOAD_MODEL']
# 	LOAD_MODEL_FILE = cfg['LOAD_MODEL_FILE']
# 	SAVE_MODEL = cfg['SAVE_MODEL']
# 	SAVE_MODEL_N = cfg['SAVE_MODEL_N']
# 	SAVE_MODEL_DIR = cfg['SAVE_MODEL_DIR']
# 	DATASET_DIR = cfg['DATASET_DIR']
# 	L_RATE = cfg['LEARNING_RATE']
# 	DECAY_RATE = cfg['DECAY_RATE']
# 	DECAY_EPOCHS = cfg['DECAY_EPOCHS']
# 	WEIGHT_DECAY = cfg['WEIGHT_DECAY']
# 	EPOCHS = cfg['EPOCHS']
# 	BATCH_SIZE = cfg['BATCH_SIZE']
# 	NUM_WORKERS = cfg['NUM_WORKERS']
# 	PIN_MEMORY = cfg['PIN_MEMORY']
# 	MIN_IMAGES = cfg['MIN_IMAGES']
#
#
# if __name__ == "__main__":
#     main()
