# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
# import cv2
import torch
from .decode_on_the_fly import _load_action_frame_nums_to_4darray
from augment.video_transformations.videotransforms import video_frames_resize
import data.base as utils
import skvideo.io

# from config import opt
def images_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class VideostreamError(BaseException):
    pass

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class VideoDataSet(data.Dataset):
    """
    3d based dataset
    random select one video and decode on the fly,
    return an array of decoded images
    the input txt file should be in format
    video_path label
    """
    def __init__(self, args, root, list_file,
                 num_segments=1,
                 new_length=64,
                 stride=1,
                 modality='rgb',
                 dataset="",
                 test_mode=False,
                 image_tmpl="",
                 transform=None,
                 full_video=None,
                 random_shift=True):

        self.root = root
        self.list_file = list_file
        self.new_length = new_length
        self.modality = modality
        self.dataset = dataset
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.stride = stride
        self._parse_list()  # get video list


    def pre_process(img):
        """change RGB [0,1] valued image to BGR [0,255]"""
        out = np.copy(img) * 255
        out = out[:, :, [2, 1, 0]]
        return out

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def get_moco_cv(self, record, random_select = True):
        """
        load all video into cv, much slow but accurate
        :param record:
        :param random_select: train:true test:false
        :return:
        """
        video_frames_num, width, height = utils.video_frame_count(self.root + record.path)
        if video_frames_num == -1:
            raise RuntimeError("No video stream avilable")
        rand_index = randint(0, max(1, video_frames_num - self.new_length * self.stride - 1))
        rand_index2 = randint(0, max(1, video_frames_num - self.new_length * self.stride - 1))
        if abs(rand_index - rand_index2) < 4 * self.stride:
            if rand_index > rand_index2:
                rand_index2 += randint(6*self.stride, max(6*self.stride, video_frames_num-3*self.stride))
            else:
                rand_index2 += randint(3*self.stride, max(6*self.stride, video_frames_num-9*self.stride))
        anchor_indexs = []
        negative_indexs = []
        for i in range(self.new_length):
            anchor_indexs.append((rand_index+i*self.stride)%video_frames_num)
            negative_indexs.append((rand_index2+i*self.stride)%video_frames_num)
        videodata = skvideo.io.vread(self.root + record.path, num_frames=max(max(anchor_indexs)+1, max(negative_indexs)+1))
        anchor = videodata[anchor_indexs, :, :, :]
        negative = videodata[negative_indexs,:,:,:]
        return anchor, negative, record.label

    def get_moco(self, record, random_select = True):
        """
        just for one segment
        :param record:
        :param random_select: train:true test:false
        :return:
        """
        #read video in this place, if no video here, random select another video
        f = open(self.root + record.path, 'rb')
        video = f.read()
        f.close()
        video_frames_num, width, height = utils.video_frame_count(self.root + record.path)
        if video_frames_num == -1:
            raise RuntimeError("No video stream avilable")
        video_frames_num -= 1 # cv2 often more 1 than ffmpeg
        rand_index = randint(0, max(1, video_frames_num - self.new_length * self.stride - 1))
        if random_select:
            decoded_images_indexs = np.arange(rand_index, self.new_length * self.stride + rand_index, self.stride)
        else:
            decoded_images_indexs = np.arange(min(rand_index, 10), min(self.new_length * self.stride + min(rand_index,10),
                                                                       video_frames_num), self.stride)
        #must be in grow seq
        decoded_images_indexs %= video_frames_num
        decoded_images_indexs = np.sort(decoded_images_indexs)
        for j in range(len(decoded_images_indexs)-1):
            if decoded_images_indexs[j] == decoded_images_indexs[j+1]:
                decoded_images_indexs[j+1] += 1
            # prevent 0,0,0,1,2,3...
            if decoded_images_indexs[j] >= decoded_images_indexs[j+1]:
                decoded_images_indexs[j+1] = decoded_images_indexs[j] + 1
        decoded_images = _load_action_frame_nums_to_4darray(video, np.sort(decoded_images_indexs), width, height)
        process_data = np.asarray(decoded_images, dtype=np.float32)

        rand_index2 = rand_index
        count = 0
        # important hyperparameter
        thresh = 4
        while abs(rand_index - rand_index2) < thresh:
            rand_index2 = randint(0, max(1, video_frames_num - self.stride * self.new_length - 1))
            count += 1
            if count > 3:
                rand_index2 = (rand_index2 + randint(max(video_frames_num//8, video_frames_num//3*2))) % video_frames_num
                break
        if random_select:
            decoded_images_indexs2 = np.arange(rand_index2, self.new_length * self.stride + rand_index2, self.stride)
        else:
            decoded_images_indexs2 = np.arange(min(rand_index2, 10), min(self.new_length + min(rand_index2,10), video_frames_num), self.stride)
        decoded_images_indexs2 %= video_frames_num
        decoded_images_indexs2 = np.sort(decoded_images_indexs2)
        #must be in grow seq
        for j in range(len(decoded_images_indexs2)-1):
            if decoded_images_indexs2[j] == decoded_images_indexs2[j+1]:
                decoded_images_indexs2[j+1] += 1
            if decoded_images_indexs2[j] >= decoded_images_indexs2[j+1]:
                decoded_images_indexs2[j+1] = decoded_images_indexs2[j] + 1
        decoded_images2 = _load_action_frame_nums_to_4darray(video, decoded_images_indexs2, width, height)
        process_data2 = np.asarray(decoded_images2, dtype=np.float32)
        return process_data, process_data2, record.label


    def get(self, record, random_select = False):
        """
        just for one segment
        :param record:
        :param random_select: train:true test:false
        :return:
        """
        #read video in this place, if no video here, random select another video
        f = open(self.root + record.path, 'rb')
        video = f.read()
        f.close()
        video_frames_num, width, height = utils.video_frame_count(self.root + record.path)
        if video_frames_num == -1:
            raise RuntimeError("No video stream avilable")
        #if video_frames_num < self.new_length:
        #    print("video{} farmes num is:{}".format(self.root + record.path, video_frames_num))
        #opencv ususlly decode more frames, so - 10 here instead of +1
        rand_index = randint(0, max(1, video_frames_num - self.new_length - 1))
        if video_frames_num > self.new_length:
            if random_select:
                decoded_images_indexs = np.arange(rand_index, self.new_length + rand_index)
            else:
                decoded_images_indexs = np.arange(min(rand_index, 10), min(self.new_length + min(rand_index,10), video_frames_num))
        else:
            decoded_images_indexs = np.arange(0, video_frames_num-1)
        #the video may be 224 x 144, need to do resize
        #if decoded_images_index is small than new_length, loop until new_length
        decoded_images = _load_action_frame_nums_to_4darray(video, decoded_images_indexs, width, height)
        if np.shape(decoded_images)[0] < self.new_length:
            for i in range(self.new_length - np.shape(decoded_images)[0]):
                decoded_images = np.concatenate((decoded_images, np.reshape(decoded_images[i%np.shape(decoded_images)[0], :, :, :], newshape=(1, height, width, 3))), axis=0)
        if np.shape(decoded_images)[0] != self.new_length:
            raise TypeError("imgs is short than need.!")
        process_data = np.asarray(decoded_images, dtype=np.float32)
        return process_data, record.label

    def __getitem__(self, index):
        '''
        i Comment all these warning message for simplify, remove this # if need.
        this part code can be simplity, but i think this way is the most straght mehod
        :param index:
        :return:
        '''
        record = self.video_list[index]
        try:
            anchor, negative, label = self.get_moco(record, random_select = not self.test_mode)
            # anchor, negative, label = self.get_moco_cv(record, random_select=not self.test_mode)
        except (IOError, ValueError, RuntimeError, TypeError, FileNotFoundError):
            #print("Error: there is no video in this place, will random select another video")
            index = randint(1, len(self.video_list))
            record = self.video_list[index]
            anchor, negative, label = self.get_moco(record, random_select= not self.test_mode)
            # anchor, negative, label = self.get_moco_cv(record, random_select=not self.test_mode)
        anchor = video_frames_resize(anchor, 256)
        anchor = 2 * (anchor / 255) - 1
        negative = video_frames_resize(negative, 256)
        negative = 2 * (negative / 255) - 1
        anchor_1 = self.transform(anchor)
        positive = self.transform(anchor)
        negative = self.transform(negative)
        return [anchor_1, positive, negative], label, index
        # record = self.video_list[index]
        # try:
        #     decoded_images, label = self.get(record, random_select = self.test_mode)
        # except (IOError, ValueError, RuntimeError, TypeError, FileNotFoundError):
        #     #print("Error: there is no video in this place, will random select another video")
        #     index = randint(1, len(self.video_list))
        #     record = self.video_list[index]
        #     decoded_images, label = self.get(record, random_select=self.test_mode)
        # decoded_images = video_frames_resize(decoded_images, 256)
        # decoded_images = 2 * (decoded_images / 255) - 1
        # decoded_images = self.transform(decoded_images)
        # decoded_images2 = self.transform(decoded_images)
        # return [decoded_images, decoded_images2], label, index

    def __len__(self):
        return len(self.video_list)