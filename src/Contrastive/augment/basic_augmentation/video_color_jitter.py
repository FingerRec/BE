import numpy as np
import random

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

class Grayscale(object):

    def __call__(self, img):
        # gs = img.clone()
        # gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        # gs[1].copy_(gs[0])
        # gs[2].copy_(gs[0])
        gs = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
        return gs


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
