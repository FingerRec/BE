import numpy as np
import numbers
import random
import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
import random
from PIL import ImageOps
# import cv2
from torchvision import transforms


class ColorDistortion(object):
    def __init__(self, s=1.0):
        self.s = s
        self.color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        self.rnd_color_jitter = transforms.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = transforms.RandomGrayscale(p=0.2)

    def __call__(self, video):
        color_distort = transforms.Compose([self.rnd_color_jitter, self.rnd_gray])
        return color_distort(video)


class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class VideoCrop(object):

    def __init__(self, size):
        self.size = size
        self.window_size = 3

    def __call__(self, imgs):
        '''
        first reshape img into 256(shorter length), then clip 3 256 x 256 img in window. if need to resize to 224 x 224 ?
        :param imgs:
        :return:
        '''
        t, h, w, c = imgs.shape # batch x 256 x 340 x 3
        th, tw = (self.size, self.size)
        video_imgs = list()
        for n in range(self.window_size):
            x1 = int(round((w - tw) / self.window_size * n))
            y1 = int(round((h - th) / self.window_size * n))
            x2 = x1 + tw
            y2 = y1 + th
            #print(x1, y1, x2, y2)
            img = np.resize(imgs[:, y1:y2, x1:x2, :], (t, th, tw, c))  # all img resize to th, tw ?
            video_imgs.append(img)
        return video_imgs

        '''
        # ===============================new version, for 256x256==========================
        t, h, w, c = imgs.shape  # batch x 256 x 340 x 3
        #print(t, h, w, c)
        th, tw = (self.size, self.size)
        video_imgs = list()
        if w > h:
            for n in range(self.window_size):
                x1 = int(round((w - h) / (self.window_size - 1) * n))
                y1 = 0
                x2 = x1 + h
                y2 = h
                #print("[{}:{},{}:{}]".format(y1, y2, x1, x2))
                img = np.zeros((t, th, tw, c))
                for i in range(t):
                    im = Image.fromarray(np.uint8((imgs[i, y1:y2, x1:x2, :] + 1) *255/2))
                    img[i] = np.asarray(im.resize((th, tw), Image.ANTIALIAS))
                img = 2 * (img / 255) - 1
                #img = np.resize(imgs[:, y1:y2, x1:x2, :], (t, th, tw, c))  # all img resize to th, tw ?
                video_imgs.append(img)
        else:
            for n in range(self.window_size):
                x1 = 0
                y1 = int(round((h - w) / (self.window_size - 1) * n))
                x2 = w
                y2 = y1 + w
                img = np.zeros((t, th, tw, c))
                for i in range(t):
                    im = Image.fromarray(np.uint8((imgs[i, y1:y2, x1:x2, :] + 1)*255/2))
                    img[i] = np.asarray(im.resize((th, tw), Image.ANTIALIAS))
                img = 2 * (img / 255) - 1
                video_imgs.append(img)
        return video_imgs
        '''
    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        corner_imgs = list()
        for n in self.crop_positions:
            #print(n)
            if n == 'c':
                th, tw = (self.size, self.size)
                x1 = int(round((w- tw) / 2.))
                y1 = int(round((h - th) / 2.))
                x2 = x1 + tw
                y2 = y1 + th
            elif n == 'tl':
                x1 = 0
                y1 = 0
                x2 = self.size
                y2 = self.size
            elif n == 'tr':
                x1 = w - self.size
                y1 = 0
                x2 = w
                y2 = self.size
            elif n == 'bl':
                x1 = 0
                y1 = h - self.size
                x2 = self.size
                y2 = h
            elif n == 'br':
                x1 = w - self.size
                y1 = h - self.size
                x2 = w
                y2 = h
            corner_imgs.append(imgs[:, y1:y2, x1:x2, :])
        return corner_imgs

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        #for t, m, s in zip(tensor, self.mean, self.std):
        #    t.sub_(m).div_(s)
        xmax, xmin = tensor.max(), tensor.min()
        tensor = (tensor - xmin) / (xmax - xmin)
        return tensor

    def randomize_parameters(self):
        pass


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        # gs = img.clone()
        # gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        # gs[1].copy_(gs[0])
        # gs[2].copy_(gs[0])
        gs = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        # return img.lerp(gs, alpha)
        cover_img = img
        for i in range(3):
            cover_img[:,:,i] = (1-alpha) * img[:,:,i] + alpha * gs
        return cover_img


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        # gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return alpha * img
        # return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        # gs = Grayscale()(img)
        # gs.fill_(gs.mean())
        # alpha = random.uniform(-self.var, self.var)
        # return img.lerp(gs, alpha)
        return np.mean(img) + self.var * (img- np.mean(img))


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, imgs):
        t, h, w, c = imgs.shape
        #print(t, h, w, c)
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        for i in range(t):
            imgs[i, :, :, :] = transform(imgs[i, :, :, :])
        return imgs


def transform_data(data, scale_size=256, crop_size=224, random_crop=False, random_flip=False):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    if random_crop:
        x0 = random.randint(0, width - crop_size)
        y0 = random.randint(0, height - crop_size)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    else:
        x0 = int((width-crop_size)/2)
        y0 = int((height-crop_size)/2)
        x1 = x0 + crop_size
        y1 = y0 + crop_size
        for i, img in enumerate(data):
            data[i] = img.crop((x0, y0, x1, y1))
    if random_flip and random.randint(0,1) == 0:
        for i, img in enumerate(data):
            data[i] = ImageOps.mirror(img)
    return  data


def get_10_crop(data, scale_size=256, crop_size=224):
    data = resize(data, scale_size)
    width = data[0].size[0]
    height = data[0].size[1]
    top_left = [[0, 0],
                [width-crop_size, 0],
                [int((width-crop_size)/2), int((height-crop_size)/2)],
                [0, height-crop_size],
                [width-crop_size, height-crop_size]]
    crop_data = []
    for point in top_left:
        non_flip = []
        flip = []
        x_0 = point[0]
        y_0 = point[1]
        x_1 = x_0 + crop_size
        y_1 = y_0 + crop_size
        for img in data:
            tmp = img.crop((x_0, y_0, x_1, y_1))
            non_flip.append(tmp)
            flip.append(ImageOps.mirror(tmp))
        crop_data.append(non_flip)
        crop_data.append(flip)
    return  crop_data


def scale(data, scale_size):
    width = data[0].size[0]
    height = data[0].size[1]
    if (width==scale_size and height>=width) or (height==scale_size and width>=height):
        return data
    if width >= height:
        h = scale_size
        w = round((width/height)*scale_size)
    else:
        w = scale_size
        h = round((height/width)*scale_size)
    for i, image in enumerate(data):
        data[i] = image.resize((w, h))
    return  data


def resize(data, scale_size):
    width = data[0].size[0]
    height = data[0].size[1]
    if (width==scale_size and height>=width) or (height==scale_size and width>=height):
        return data
    for i, image in enumerate(data):
        data[i] = image.resize((scale_size, scale_size))
    return  data


def video_frames_resize(data, scale_size):
    t, h, w, c = data.shape

    if h >= scale_size and w >= scale_size:
        return data
    else:
        data2 = data.copy()
        data2.resize((t, scale_size, scale_size, c))
        return data2

