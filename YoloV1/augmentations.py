from random import  uniform
from PIL.JpegImagePlugin import JpegImageFile
from PIL.Image import Image
import torch
from torchvision import transforms as T

class RTranslation(object):
    """
    Randomly scales images by not more than "scale" ratio. By default scale = 0.2.
    Scaling is done from the (0,0) point (left/top point)

    By default scaling is done towards diagonal (same x_ratio and y_ratio) but implemented separately towards x and y
    """
    def __init__(self, dist = 0.2, diff = False):
        self.dist = dist
        if type(self.dist) in [list, tuple]:
            assert len(self.dist) == 2, 'R.SCALING: Problems with input of '
            self.x, self.y = scale
        else:
            assert type(scale) == float, 'R.SCALING: The type of scale is not float'
            self.x = self.y = scale
        self.diff = diff

    def __call__(self, image, bboxes):
        if self.diff:
            scale_x, scale_y = 1 - uniform(0, self.x), 1 - uniform(0, self.y)
        else:
            scale_x = scale_y = 1 - uniform(0, self.x)

        if type(image) == torch.Tensor:
            image = T.ToPILImage()(image)
        assert type(image) in (JpegImageFile, Image), 'R.SCALING: Problems with type of the image!'

        w, h = image.size
        new_w, new_h = int(w * scale_x), int(h * scale_y)
        image = T.Resize((new_h, new_w))(image)
        # print('1:', type(image), image.size,  new_w, new_h)

        new_image = torch.zeros(3, h, w)
        # print('2:', type(T.ToTensor()(image)), T.ToTensor()(image).shape, h, w, new_h, new_w)
        new_image[:,:new_h, :new_w] = T.ToTensor()(image)

        bboxes[:, 1] = bboxes[:, 1] * scale_x
        bboxes[:, 3] = bboxes[:, 3] * scale_x
        bboxes[:, 2] = bboxes[:, 2] * scale_y
        bboxes[:, 4] = bboxes[:, 4] * scale_y
        return T.ToPILImage()(new_image), bboxes

class RScaling(object):
    """
    Randomly scales images by not more than "scale" ratio. By default scale = 0.2.
    Scaling is done from the (0,0) point (left/top point)

    By default scaling is done towards diagonal (same x_ratio and y_ratio) but implemented separately towards x and y
    """
    def __init__(self, scale = 0.2, diff = False):
        self.scale = scale
        if type(scale) in [list, tuple]:
            assert len(scale) == 2, 'R.SCALING: Problems with input of '
            self.x, self.y = scale
        else:
            assert type(scale) == float, 'R.SCALING: The type of scale is not float'
            self.x = self.y = scale
        self.diff = diff

    def __call__(self, image, bboxes):
        if self.diff:
            scale_x, scale_y = 1 - uniform(0, self.x), 1 - uniform(0, self.y)
        else:
            scale_x = scale_y = 1 - uniform(0, self.x)

        if type(image) == torch.Tensor:
            image = T.ToPILImage()(image)
        assert type(image) in (JpegImageFile, Image), 'R.SCALING: Problems with type of the image!'

        w, h = image.size
        new_w, new_h = int(w * scale_x), int(h * scale_y)
        image = T.Resize((new_h, new_w))(image)
        # print('1:', type(image), image.size,  new_w, new_h)

        new_image = torch.zeros(3, h, w)
        # print('2:', type(T.ToTensor()(image)), T.ToTensor()(image).shape, h, w, new_h, new_w)
        new_image[:,:new_h, :new_w] = T.ToTensor()(image)

        bboxes[:, 1] = bboxes[:, 1] * scale_x
        bboxes[:, 3] = bboxes[:, 3] * scale_x
        bboxes[:, 2] = bboxes[:, 2] * scale_y
        bboxes[:, 4] = bboxes[:, 4] * scale_y
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
    def __init__(self, saturation = 0.5, brightness = 0.5):
        self.saturation = saturation
        self.brightness = brightness

    def __call__(self, image, bboxes):
        if type(image) == torch.Tensor:
            image = T.ToPILImage()(image)
        assert type(image) in (JpegImageFile, Image), 'R.H.FLIP: Problems with type of the image: {}'.format(type(image))
        image = T.ColorJitter(saturation = self.saturation, brightness = self.brightness)(image) # TODO: what about contrast and hue?
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
        RScaling(),
        RHorizontalFlip(),
        ColorTransforms(),

    # transforms.ColorJitter(brightness=0.5, hue=.05, saturation=.05),
    ]),
    'val': Compose([])
}

