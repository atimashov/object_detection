import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.init import kaiming_normal_
from torch import nn
import torch.optim as optim
from termcolor import colored
import os
import json
from dataset import VOCDataset
from models import Darknet, YoloV1_pretrained, OD_backbone

def init_weights(model):
    for layer in model.features:
        if type(layer) in [nn.Conv2d, nn.Linear]:
            kaiming_normal_(layer.weight)
    for layer in model.classifier:
        if type(layer) in [nn.Conv2d, nn.Linear]:
            kaiming_normal_(layer.weight)
    return model # TODO: should I return?

def get_dataloader(params, transform):
    # create data loader
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DATA_DIR, CSV_TRAIN, CSV_VAL = params
    print('***:',type(CSV_TRAIN),type(CSV_VAL))
    img_dir = '{}/images'.format(DATA_DIR)
    label_dir = '{}/labels'.format(DATA_DIR)
    # create annotations
    if type(CSV_TRAIN) == str:
        csv_t = pd.read_csv('{}/{}'.format(DATA_DIR, CSV_TRAIN))
    else:
        csv_t = pd.DataFrame(columns=['image', 'annotation'])
        for csv_dir in CSV_TRAIN:
            csv_t = csv_t.append(pd.read_csv('{}/{}'.format(DATA_DIR, csv_dir)), ignore_index=True)

    if type(CSV_VAL) == str:
        csv_v = pd.read_csv('{}/{}'.format(DATA_DIR, CSV_VAL))
    else:
        csv_v = pd.DataFrame(columns=['image', 'annotation'])
        for csv_dir in CSV_VAL:
            csv_v = csv_v.append(pd.read_csv('{}/{}'.format(DATA_DIR, csv_dir)), ignore_index=True)

    my_transform = transform

    data_train = VOCDataset(
        dataset_csv = csv_t, img_dir = img_dir, label_dir = label_dir , transform = my_transform['train']
    )
    data_val = VOCDataset(
        dataset_csv = csv_v, img_dir = img_dir, label_dir = label_dir , transform = my_transform['val']
    )
    data_loader = {
        'train': DataLoader(
            data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
            drop_last=True, pin_memory=PIN_MEMORY
        ),
        'val': DataLoader(
            data_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
            drop_last=True, pin_memory=PIN_MEMORY
        )
    }
    return data_loader

def get_optimizer(optimizer, model, cfg, epoch, lr):
    flag = cfg['scheduler']['flag']
    epochs = cfg['scheduler']['epochs'] if flag else [i * cfg['decay']['epochs'] for i in range(100)]
    wd = cfg['weight_decay']

    if epoch not in epochs:
        return optimizer, cfg['name'], lr
    # it is time to change optimizer
    if flag:
        lrs = {epochs[i]:cfg['scheduler']['lr'][i] for i in range(len(epochs))}
        lr = lrs[epoch]
    else:
        p = (epoch + 1) //  cfg['decay']['epochs']
        lr = cfg['lr'] * (1 - cfg['decay']['rate']) ** p
    if cfg['name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    elif cfg['name'] == 'SGD+Nesterov':
        momentum = cfg['momentum']
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay = wd
        )
    else: # TODO: change to None
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    return optimizer, cfg['name'], lr

def create_stat(stat_path, cfg):
    files = os.listdir(stat_path)
    if 'stat.csv' not in files:
        cols = [
                'number_id', 'date_time', 'epoch', 'loss_train', 'loss_validate',
                'maps_train', 'maps_validate'
        ]
        data = pd.DataFrame(columns = cols)
        data.to_csv('{}/stat.csv'.format(stat_path), index = False)
    if 'config.json' not in files:
        nid = 1
        data = {str(nid):cfg}
    else:
        with open('{}/config.json'.format(stat_path)) as json_file:
            data = json.load(json_file)
        nid = max(map(int, data.keys())) + 1
        data[str(nid)] = cfg

    with open('{}/config.json'.format(stat_path), 'w') as outfile:
        json.dump(data, outfile)
    return nid


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format = 'midpoint', num_classes=20
):
    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """
    predictions = predictions.to("cpu") # what if to change to GPU / remove this row
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1) # .to(torch.device('cuda:1'))
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

def convert_cellboxes_new(predictions, S = 7, B = 2, C = 20):
    # predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + B * 5)

    # TODO: rewrite for the case when B <> 2
    bboxes1 = predictions[..., (C + 5 * 0 + 1):(C + 5 * 0 + 5)]
    bboxes2 = predictions[..., (C + 5 * 1 + 1):(C + 5 * 1 + 5)]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim = 0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., 0:1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3)) # TODO: is it correct?
    w_h = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_h), dim = -1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5 * 1]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim = -1
    )
    return converted_preds



def cellboxes_to_boxes(out, S=7): # TODO: I don't need this function
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    # print('SASHA')
    # print(converted_pred[0,0,:])
    # print()
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader, model, iou_threshold, threshold, device, box_format = 'midpoint'
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        print('labels', labels.size())
        print('predictions', predictions.size())
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def get_metrics_NEW(
    loader, model, iou_threshold, threshold, device, loss_func, box_format = 'midpoint',
        S = 7, B = 2, C = 20
):
    pred_bb = []
    target_bb = []
    losses = []
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(imgs)

        loss = loss_func(predictions, labels)
        losses.append(float(loss))

        batch_size = imgs.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                pred_bb.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    target_bb.append([train_idx] + box)

            train_idx += 1

    # loss
    loss = sum(losses) / len(losses)
    # mAP
    maps = []
    for iou in np.arange(0.5, 1, 0.05):
        maps.append(
            float(mean_average_precision(pred_bb, target_bb, iou_threshold=iou, box_format='midpoint'))
        )

    model.train()
    return loss, maps



def get_metrics(
    loader, model, iou_threshold, threshold, device, dtype, loss_func, S = 7, B = 2, C = 20
):
    pred_bb, target_bb = get_bboxes(
        loader = loader, model = model, iou_threshold = iou_threshold, threshold = threshold,
        device= device
    )

    # loss function
    print('*', len(pred_bb), len(target_bb))
    print('**', pred_bb[0].shape)
    t_pred_bb = torch.cat(pred_bb, dim = 0)
    print('***', t_pred_bb.shape)
    t_target_bb = torch.cat(target_bb, dim = 0)
    loss = loss_func(t_pred_bb, t_target_bb)

    # mAP
    maps = []
    for iou in np.arange(0.5, 1, 0.05):
        maps.append(
            float(mean_average_precision(pred_bb, target_bb, iou_threshold = iou, box_format = 'midpoint'))
        )
    print('get metrics: ', type(loss), type(maps[0]))
    return loss, maps

def print_report(part, epoch = None, t = None, metrics = None):
    if part == 'start':
        print('{} Epoch {}{} {}'.format(' ' * 60, ' ' * (3 - len(str(epoch))), epoch, ' ' * 61))
        print(' ' * 132)
    elif part == 'end':
        t_min, t_sec = str(t // 60), str(t % 60)
        txt = 'It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec)
        print(txt)
        print()
        print(colored('-' * 132, 'cyan'))
        print()
    else: # report statistics
        # loss
        train_loss, val_loss, train_maps, val_maps = metrics
        t_loss, v_loss = round(train_loss, 3), round(val_loss, 3)
        prefix = 6 - str(t_loss).find('.')
        postfix = 4 - (len(str(t_loss)) - str(t_loss).find('.'))
        t_loss = '{}{}{}'.format(' ' * prefix, t_loss, '0' * postfix)
        prefix = 6 - str(v_loss).find('.')
        postfix = 4 - (len(str(v_loss)) - str(v_loss).find('.'))
        v_loss = '{}{}{}'.format(' ' * prefix, v_loss, '0' * postfix)
        t_print = 'TRAIN   : loss = {}'.format(t_loss)
        v_print = 'VALIDATE: loss = {}'.format(v_loss)

        # mAP@0.5
        t_map, v_map = round(100 * train_maps[0], 2), round(100 * val_maps[0], 2)
        if '.' not in str(t_map):
            t_map = '{}{}.00'.format((3 - len(str(t_map))) * ' ' , t_map)
        else:
            prefix = 3 - str(t_map).find('.')
            postfix = 2 - (len(str(t_map)) - str(t_map).find('.'))
            t_map = '{}{}{}'.format(' ' * prefix, t_map, '0' * postfix)

        if '.' not in str(v_map):
            v_map = '{}{}.00'.format((3 - len(str(v_map))) * ' ', v_map)
        else:
            prefix = 3 - str(v_map).find('.')
            postfix = 2 - (len(str(v_map)) - str(v_map).find('.'))
            v_map = '{}{}{}'.format(' ' * prefix, v_map, '0' * postfix)

        t_print += ' | mAP@0.5  = {}%'.format(t_map) # mAP@0.5:0.05:0.95
        v_print += ' | mAP@0.5  = {}%'.format(v_map) # mAP@0.5:0.05:0.95

        # mAP@0.5:0.05:0.95
        t_map, v_map = round(100 * np.mean(train_maps), 2), round(100 * np.mean(val_maps), 2)
        if '.' not in str(t_map):
            t_map = '{}{}.00'.format((3 - len(str(t_map))) * ' ', t_map)
        else:
            prefix = 3 - str(t_map).find('.')
            postfix = 2 - (len(str(t_map)) - str(t_map).find('.'))
            t_map = '{}{}{}'.format(' ' * prefix, t_map, '0' * postfix)

        if '.' not in str(v_map):
            v_map = '{}{}.00'.format((3 - len(str(v_map))) * ' ', v_map)
        else:
            prefix = 3 - str(v_map).find('.')
            postfix = 2 - (len(str(v_map)) - str(v_map).find('.'))
            v_map = '{}{}{}'.format(' ' * prefix, v_map, '0' * postfix)

        t_print += ' | mAP@0.5:0.05:0.95  = {}%'.format(t_map)  # mAP@0.5:0.05:0.95
        v_print += ' | mAP@0.5:0.05:0.95  = {}%'.format(v_map)  # mAP@0.5:0.05:0.95

        print(t_print)
        print(v_print)

def get_n_classes(min_class, root = 'home/alex/datasets/imagenet/'):
    classes = os.listdir(root)
    out = 0
    for my_class in classes:
        if '.' in my_class: continue
        imgs = os.listdir('{}/{}'.format(root, my_class))
        out +=  (len(imgs) >= min_class)
    return out

def load_checkpoint(file_name, device, S, B, C, cfg = None):
    print('Loading checkpoint...')
    if cfg is None:
        checkpoint = torch.load(file_name)
        cfg = checkpoint['cfg']
    BACKBONE = cfg['MODEL']
    N_LAYERS = cfg.get('N_LAYERS', 0)
    MIN_IMAGES = cfg['MIN_IMAGES']
    DATASET_DIR = cfg['DATASET_DIR']
    print(MIN_IMAGES, DATASET_DIR)

    print('backbone:', BACKBONE)
    if BACKBONE == 'Darknet':
        n_classes = get_n_classes(MIN_IMAGES, root=DATASET_DIR)
        backbone = Darknet(n_layers = N_LAYERS, num_classes = n_classes)
        backbone.load_state_dict(checkpoint['model'])
        model = YoloV1_pretrained(
            backbone = backbone, n_layers=N_LAYERS, grid_size=S, num_boxes=B, num_classes=C
        ).to(device)

    # for param in backbone.parameters():
    #     param.requires_grad = False

    elif BACKBONE == 'VGG16':
        model = OD_backbone(
            bb_name = BACKBONE, grid_size = S, num_boxes = B, num_classes = C
        ).to(device)

    if BACKBONE == 'Darknet':
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    return model, cfg, epoch


