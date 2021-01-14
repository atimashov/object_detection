from random import  uniform
from PIL.JpegImagePlugin import JpegImageFile
from PIL.Image import Image
import torch
from torchvision import transforms as T
from tqdm import tqdm
from dataset import VOCDataset
from torch.utils.data import DataLoader
import cv2
from torchvision.utils import  save_image

class RTranslation(object):
    """
    Randomly scales images by not more than "scale" ratio. By default scale = 0.2.
    Scaling is done from the (0,0) point (left/top point)

    By default scaling is done towards diagonal (same x_ratio and y_ratio) but implemented separately towards x and y

    bboxes: class_label, x, y, width, height
    """
    def __init__(self, translate = 0.2, diff = False, thresh = 0.2):
        if type(translate) in [list, tuple]:
            assert len(translate) == 2, 'R.SCALING: Problems with input of '
            self.x, self.y = translate
        else:
            assert type(translate) == float, 'R.SCALING: The type of scale is not float'
            self.x = self.y = translate
        self.thresh = thresh
        self.diff = diff

    def clip(self):
        pass

    def __call__(self, image, bboxes):
        if self.diff:
            trans_x, trans_y = uniform(-self.x, self.x), uniform(-self.y, self.y)
        else:
            trans_x = trans_y = uniform(-self.x, self.x)

        if type(image) == torch.Tensor:
            image = T.ToPILImage()(image)
        assert type(image) in (JpegImageFile, Image), 'R.SCALING: Problems with type of the image!'

        # create image
        w, h = image.size
        x1_from, x2_from = int(max(0, trans_x) * w), int(min(1, 1 + trans_x) * w)
        x1_to, x2_to = w - x2_from, w - x1_from
        y1_from, y2_from = int(max(0, trans_y) * h), int(min(1, 1 + trans_y) * h)
        y1_to, y2_to = h - y2_from, h - y1_from
        # print(w, h, '||', x1_from, x2_from, y1_from, y2_from, '||', x1_to, x2_to, y1_to, y2_to)
        image = image.crop((x1_from, y1_from, x2_from, y2_from))
        # print(image.size)
        new_image = torch.zeros(3, h, w)
        new_image[:, y1_to:y2_to, x1_to:x2_to] = T.ToTensor()(image)

        # create bboxes
        try:
            # print(1)
            init_area = bboxes[:,3] * bboxes[:,4]
        except:
            save_image(T.ToTensor()(image), 'error_img.jpeg')
            print('+++', bboxes.shape)
            init_area = bboxes[:,3] * bboxes[:,4]

        x1 = bboxes[:, 1] - bboxes[:, 3] / 2 - trans_x
        x2 = bboxes[:, 1] + bboxes[:, 3] / 2 - trans_x

        x1[x1 < 0], x2[x2 < 0] = 0, 0
        x1[x1 > 1], x2[x2 > 1] = 1, 1

        y1 = bboxes[:, 2] - bboxes[:, 4] / 2 - trans_y
        y2 = bboxes[:, 2] + bboxes[:, 4] / 2 - trans_y
        y1[y1 < 0], y2[y2 < 0] = 0, 0
        y1[y1 > 1], y2[y2 > 1] = 1, 1

        bboxes[:, 1], bboxes[:, 2] = (x1 + x2) / 2, (y1 + y2) / 2
        bboxes[:, 3], bboxes[:, 4] = x2-x1, y2-y1
        new_area = bboxes[:,3] * bboxes[:,4]
        bboxes = bboxes[new_area/init_area > self.thresh]

        return T.ToPILImage()(new_image), bboxes

class RScaling(object):
    """
    Randomly scales images by not more than "scale" ratio. By default scale = 0.2.
    Scaling is done from the (0,0) point (left/top point)

    By default scaling is done towards diagonal (same x_ratio and y_ratio) but implemented separately towards x and y
    """
    def __init__(self, scale = 0.2, diff = False, thresh = 0.2):
        self.scale = scale
        if type(scale) in [list, tuple]:
            assert len(scale) == 2, 'R.SCALING: Problems with input of '
            self.x, self.y = scale
        else:
            assert type(scale) == float, 'R.SCALING: The type of scale is not float'
            self.x = self.y = scale
        self.thresh = thresh
        self.diff = diff

    def __call__(self, image, bboxes):
        if self.diff:
            scale_x, scale_y = 1 + uniform(-self.x, self.x), 1 + uniform(-self.y, self.y)
        else:
            scale_x = scale_y = 1 + uniform(-self.x, self.x)

        if type(image) == torch.Tensor:
            image = T.ToPILImage()(image)
        assert type(image) in (JpegImageFile, Image), 'R.SCALING: Problems with type of the image!'

        w, h = image.size
        new_w, new_h = int(w * scale_x), int(h * scale_y)
        image = T.Resize((new_h, new_w))(image)
        # print('1:', type(image), image.size,  new_w, new_h)

        new_image = torch.zeros(3, h, w)
        # print('2:', type(T.ToTensor()(image)), T.ToTensor()(image).shape, h, w, new_h, new_w)
        new_image[:,:min(new_h, h), :min(new_w, w)] = T.ToTensor()(image)[:,:min(new_h, h), :min(new_w, w)]

        # create bboxes
        init_area = bboxes[:, 3] * bboxes[:, 4]
        bboxes[:, 1] = bboxes[:, 1] * scale_x
        bboxes[:, 3] = bboxes[:, 3] * scale_x
        bboxes[:, 2] = bboxes[:, 2] * scale_y
        bboxes[:, 4] = bboxes[:, 4] * scale_y

        x1 = bboxes[:, 1] - bboxes[:, 3] / 2
        x2 = bboxes[:, 1] + bboxes[:, 3] / 2
        x1[x1 < 0], x2[x2 < 0] = 0, 0
        x1[x1 > 1], x2[x2 > 1] = 1, 1

        y1 = bboxes[:, 2] - bboxes[:, 4] / 2
        y2 = bboxes[:, 2] + bboxes[:, 4] / 2
        y1[y1 < 0], y2[y2 < 0] = 0, 0
        y1[y1 > 1], y2[y2 > 1] = 1, 1

        bboxes[:, 1], bboxes[:, 2] = (x1 + x2) / 2, (y1 + y2) / 2
        bboxes[:, 3], bboxes[:, 4] = x2 - x1, y2 - y1
        new_area = bboxes[:, 3] * bboxes[:, 4]
        bboxes = bboxes[new_area / init_area > self.thresh]

        return T.ToPILImage()(new_image), bboxes

class RHorizontalFlip(object):
    """
    Randomly scales images by not more than "scale" ratio. By default scale = 0.2.
    Scaling is done from the (0,0) point (left/top point)

    By default scaling is done towards diagonal (same x_ratio and y_ratio) but implemented separately towards x and y
    """
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, bboxes):
        if type(image) == torch.Tensor:
            image = T.ToPILImage()(image)
        assert type(image) in (JpegImageFile, Image), 'R.H.FLIP: Problems with type of the image: {}'.format(type(image))
        flip = uniform(0, 1) < self.prob
        if flip:
            image = T.RandomHorizontalFlip(1)(image)
            bboxes[:, 1], bboxes[:, 2] = bboxes[:, 2], bboxes[:, 1]
        return image, bboxes

class ColorTransforms(object):
    """
    Randomly scales images by not more than "scale" ratio. By default scale = 0.2.
    Scaling is done from the (0,0) point (left/top point)

    By default scaling is done towards diagonal (same x_ratio and y_ratio) but implemented separately towards x and y
    """
    def __init__(self, saturation = 0.5, brightness = 0.5, hue = 0.5, contrast = 0.5):
        self.saturation = saturation
        self.brightness = brightness
        self.hue = hue
        self.contrast = contrast

    def __call__(self, image, bboxes):
        if type(image) == torch.Tensor:
            image = T.ToPILImage()(image)
        assert type(image) in (JpegImageFile, Image), 'R.H.FLIP: Problems with type of the image: {}'.format(type(image))
        image = T.ColorJitter(
            saturation = self.saturation, brightness = self.brightness,
            hue = self.hue, contrast = self.contrast
        )(image) # TODO: what about contrast and hue?
        return image, bboxes

class BlurTransforms(object):
    """
    """
    def __init__(self, sigma = 0.5):
        self.sigma = sigma

    def __call__(self, image, bboxes):
        if type(image) == torch.Tensor:
            image = T.ToPILImage()(image)
        assert type(image) in (JpegImageFile, Image), 'R.H.FLIP: Problems with type of the image: {}'.format(type(image))
        image = T.GaussianBlur(kernel_size = 3, sigma=(0.1, 2.0))(image) # TODO: what about contrast and hue?
        return image, bboxes


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        img = T.Resize((448, 448))(img)
        img = T.ToTensor()(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img) # TODO: check for PascalVOC specifically

        return img, bboxes

my_transforms = {
    'train': Compose([
        RTranslation(),
        RScaling(),
        RHorizontalFlip(),
        # BlurTransforms(),
        ColorTransforms(saturation=0.2, brightness=0.2),
    ]),
    'val': Compose([])
}

def rectangle(t, x1, x2, y1, y2, thickness = 3, color = (255, 0, 0)):
    R, G, B = color
    t[0, (y1 + thickness - 1):(y2 + 1), x1:(x1 + thickness)] = R
    t[1, (y1 + thickness - 1):(y2 + 1), x1:(x1 + thickness)] = G
    t[2, (y1 + thickness - 1):(y2 + 1), x1:(x1 + thickness)] = B

    t[0, (y1 + thickness - 1):(y2 + 1), (x2-thickness + 1):(x2 + 1)] = R
    t[1, (y1 + thickness - 1):(y2 + 1), (x2-thickness + 1):(x2 + 1)] = G
    t[2, (y1 + thickness - 1):(y2 + 1), (x2-thickness + 1):(x2 + 1)] = B

    t[0, y1:(y1 + thickness), (x1 + thickness -1):(x2 + 1)] = R
    t[1, y1:(y1 + thickness), (x1 + thickness -1):(x2 + 1)] = G
    t[2, y1:(y1 + thickness), (x1 + thickness -1):(x2 + 1)] = B

    t[0, (y2 - thickness + 1):(y2 + 1), (x1 + thickness -1):(x2 + 1)] = R
    t[1, (y2 - thickness + 1):(y2 + 1), (x1 + thickness -1):(x2 + 1)] = G
    t[2, (y2 - thickness + 1):(y2 + 1), (x1 + thickness -1):(x2 + 1)] = B
    return t


def test(target_dir = 'augmented'):
    data_dir = '/home/alex/datasets/PascalVOC'
    img_dir = '{}/images'.format(data_dir)
    label_dir = '{}/labels'.format(data_dir)
    csv_dir = '{}/8examples.csv'.format(data_dir)
    # csv_dir = '{}/1example.csv'.format(data_dir)

    # init transform
    transform = Compose([])
    data = VOCDataset(
        dataset_csv = csv_dir, img_dir = img_dir, label_dir = label_dir , transform = transform, test = True
    )
    for img_id, (img, labels) in enumerate(data):
        w, h = img.shape[1:]
        for j in range(labels.shape[0]):
            x1, x2 = int((labels[j, 1] - labels[j, 3] / 2) * w), int((labels[j, 1] + labels[j, 3] / 2) * w)
            y1, y2 = int((labels[j, 2] - labels[j, 4] / 2) * h), int((labels[j, 2] + labels[j, 4] / 2) * h)
            img = rectangle(img, x1, x2, y1, y2)
            # save_image(img, '{}/{}_bb{}.jpeg'.format(target_dir, img_id, j))
        save_image(img, '{}/{}_bb.jpeg'.format(target_dir, img_id))

    # translation transform
    transform = Compose([RTranslation(),])
    data = VOCDataset(
        dataset_csv = csv_dir, img_dir = img_dir, label_dir = label_dir , transform = transform, test = True
    )
    for img_id, (img, labels) in enumerate(data):
        w, h = img.shape[1:]
        for j in range(labels.shape[0]):
            x1, x2 = int((labels[j, 1] - labels[j, 3] / 2) * w), int((labels[j, 1] + labels[j, 3] / 2) * w)
            y1, y2 = int((labels[j, 2] - labels[j, 4] / 2) * h), int((labels[j, 2] + labels[j, 4] / 2) * h)
            img = rectangle(img, x1, x2, y1, y2)
            # save_image(img, '{}/{}_bb{}.jpeg'.format(target_dir, img_id, j))
        save_image(img, '{}/{}_transl.jpeg'.format(target_dir, img_id))

    # scaling transform
    transform = Compose([RScaling(),])
    data = VOCDataset(
        dataset_csv = csv_dir, img_dir = img_dir, label_dir = label_dir , transform = transform, test = True
    )
    for img_id, (img, labels) in enumerate(data):
        w, h = img.shape[1:]
        for j in range(labels.shape[0]):
            x1, x2 = int((labels[j, 1] - labels[j, 3] / 2) * w), int((labels[j, 1] + labels[j, 3] / 2) * w)
            y1, y2 = int((labels[j, 2] - labels[j, 4] / 2) * h), int((labels[j, 2] + labels[j, 4] / 2) * h)
            img = rectangle(img, x1, x2, y1, y2)
            # save_image(img, '{}/{}_bb{}.jpeg'.format(target_dir, img_id, j))
        save_image(img, '{}/{}_scale.jpeg'.format(target_dir, img_id))


def test2(target_dir = 'augmented2'):
    data_dir = '/home/alex/datasets/PascalVOC'
    img_dir = '{}/images'.format(data_dir)
    label_dir = '{}/labels'.format(data_dir)
    csv_dir = '{}/8examples.csv'.format(data_dir)
    # csv_dir = '{}/1example.csv'.format(data_dir)

    # init transform
    transform = Compose([])
    data = VOCDataset(
        dataset_csv = csv_dir, img_dir = img_dir, label_dir = label_dir , transform = transform, test = True
    )
    for img_id, (img, labels) in enumerate(data):
        w, h = img.shape[1:]
        for j in range(labels.shape[0]):
            x1, x2 = int((labels[j, 1] - labels[j, 3] / 2) * w), int((labels[j, 1] + labels[j, 3] / 2) * w)
            y1, y2 = int((labels[j, 2] - labels[j, 4] / 2) * h), int((labels[j, 2] + labels[j, 4] / 2) * h)
            img = rectangle(img, x1, x2, y1, y2)
            # save_image(img, '{}/{}_bb{}.jpeg'.format(target_dir, img_id, j))
        save_image(img, '{}/{}_bb.jpeg'.format(target_dir, img_id))


    # translation transform
    transform = Compose([RTranslation(),])
    data = VOCDataset(
        dataset_csv = csv_dir, img_dir = img_dir, label_dir = label_dir , transform = transform, test = True
    )
    for epoch in range(5):
        for img_id, (img, labels) in enumerate(data):
            w, h = img.shape[1:]
            for j in range(labels.shape[0]):
                x1, x2 = int((labels[j, 1] - labels[j, 3] / 2) * w), int((labels[j, 1] + labels[j, 3] / 2) * w)
                y1, y2 = int((labels[j, 2] - labels[j, 4] / 2) * h), int((labels[j, 2] + labels[j, 4] / 2) * h)
                img = rectangle(img, x1, x2, y1, y2)
                # save_image(img, '{}/{}_bb{}.jpeg'.format(target_dir, img_id, j))
            save_image(img, '{}/{}_transl{}.jpeg'.format(target_dir, img_id, epoch))


    # scaling transform
    transform = Compose([RScaling(),])
    data = VOCDataset(
        dataset_csv = csv_dir, img_dir = img_dir, label_dir = label_dir , transform = transform, test = True
    )
    for epoch in range(5):
        for img_id, (img, labels) in enumerate(data):
            w, h = img.shape[1:]
            for j in range(labels.shape[0]):
                x1, x2 = int((labels[j, 1] - labels[j, 3] / 2) * w), int((labels[j, 1] + labels[j, 3] / 2) * w)
                y1, y2 = int((labels[j, 2] - labels[j, 4] / 2) * h), int((labels[j, 2] + labels[j, 4] / 2) * h)
                img = rectangle(img, x1, x2, y1, y2)
                # save_image(img, '{}/{}_bb{}.jpeg'.format(target_dir, img_id, j))
            save_image(img, '{}/{}_scale{}.jpeg'.format(target_dir, img_id, epoch))



if __name__=='__main__':
	test2()

