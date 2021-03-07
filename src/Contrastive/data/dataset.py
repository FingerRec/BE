# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random
from data.base import *
from augment.optical_flow.warp import warp_flow
from augment.basic_augmentation.spatial_wrap import spatial_warp_torch


class DataSet(data.Dataset):
    def __init__(self, args, root_path, list_file, dataset='ucf101',
                 num_segments=1, new_length=64, stride=1, modality='rgb',
                 image_tmpl='img_{:06d}.jpg', transform=None,
                 random_shift=True, test_mode=False, full_video=False):
        self.args = args
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.stride = stride
        self.modality = modality
        self.dataset = dataset
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.full_video = full_video
        # if self.args.eval_indict == 'loss':
        #     self.clips = int(args.clips)
        #     self.clip_length = new_length // int(self.clips)
        if self.test_mode:
            self.test_frames = 250
        self._parse_list()  # get video list
        # self.new_length = 150
        # data augmentation

    def _load_image(self, directory, idx):
        # if self.dataset == 'diving48':
        #     directory = directory[:]
        directory = self.root_path + directory
        if self.modality == 'rgb' or self.modality == 'RGBDiff' or self.modality == 'RGB':
            if self.dataset == 'hmdb51':
                #directory = "/data1/DataSet/Hmdb51/person/" + \
                #directory = "/data1/DataSet/Hmdb51/hmdb51/" + \
                #directory = "/data1/DataSet/hmdb51_sta_frames2/" + \
                directory = "/data1/DataSet/Hmdb51/hmdb51/" + \
                            directory.strip().split(' ')[0].split('/')[-1]
            elif self.dataset == 'ucf101':
                directory = "/data1/DataSet/UCF101/jpegs_256/" + \
                            directory.strip().split(' ')[0].split('/')[-1]
            else:
                Exception("wrong dataset!")
            # img = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')
            try:
                img = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')
            except (IOError, ValueError, RuntimeError, TypeError, FileNotFoundError):
                # print("lack image in: {}".format(directory))
                img = Image.open(os.path.join(directory, self.image_tmpl.format(1))).convert('RGB')
            # width, height = img.size
            # if width < 256 or height < 256:
            #     # print(width, height)
            #     img = img.resize((max(256, int(256 / height * width)), 256), Image.BILINEAR)
            # if self.args.spatial_size == '112':
            #     width, height = img.size
            #     img = img.resize((int(128 / height * width), 128), Image.BILINEAR)
            return [img]
        elif self.modality == 'flow':
            if self.dataset == 'hmdb51':
                directory = "/data/home/awinywang/Data/ft_local/hmdb51/tvl1_flow/{}/" + \
                            directory.strip().split(' ')[0].split('/')[-1]
            elif self.dataset == 'ucf101':
                directory = "/data/home/awinywang/Data/ft_local/ucf101/tvl1_flow/{}/" + \
                            directory.strip().split(' ')[0].split('/')[-1]
            else:
                Exception("wrong dataset!")
            u_img_path = directory.format('u') + '/frame' + str(idx).zfill(6) + '.jpg'
            v_img_path = directory.format('v') + '/frame' + str(idx).zfill(6) + '.jpg'
            x_img = Image.open(u_img_path).convert('L')
            y_img = Image.open(v_img_path).convert('L')
            return [x_img, y_img]

    def _load_gen_image(self, directory, idx, prob=1):
        directory = self.root_path + directory
        if self.dataset == 'hmdb51':
            directory = "/data1/awinywang/hmdb51/jpegs_256/" + directory.strip().split(' ')[0].split('/')[-1]
        elif self.dataset == 'ucf101':
            directory = "/data1/awinywang/ucf101/jpegs_256/" + directory.strip().split(' ')[0].split('/')[-1]
        else:
            Exception("wrong dataset!")
        rgb_img = cv2.imread(os.path.join(directory, self.image_tmpl.format(idx)))
        if self.dataset == 'ucf101':
            u_img_path = directory + '/frame' + str(idx).zfill(6) + '.jpg'
            v_img_path = directory + '/frame' + str(idx).zfill(6) + '.jpg'
        else:
            u_img_path = os.path.join(directory, self.image_tmpl.format('flow_x', idx))
            v_img_path = os.path.join(directory, self.image_tmpl.format('flow_y', idx))
        x_img = cv2.imread(u_img_path)
        y_img = cv2.imread(v_img_path)
        flow = np.zeros((x_img.shape[0], x_img.shape[1], 2), dtype=np.float32)
        flow[:, :, 0] = x_img[:, :, 0]
        flow[:, :, 1] = y_img[:, :, 0]
        # prob = max(10, np.random.random() * 60)
        prob = max(0.01, np.random.random() * 10)
        norm_flow = np.zeros(flow.shape, dtype=np.float32)
        # mask = np.zeros(flow.shape, dtype=np.float32)
        # mask[norm_flow[:, :] > np.mean(norm_flow)] = 1
        cv2.normalize(flow, dst=norm_flow, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        gen_img = warp_flow(rgb_img, norm_flow * prob)
        # gen_img = warp_flow(rgb_img, (flow / 255 - 1) * prob)
        rgb_img = Image.fromarray(np.uint8(rgb_img))
        gen_img = Image.fromarray(np.uint8(gen_img))
        return [rgb_img], [gen_img]

    def _load_gen_image_2(self, directory, idx):
        directory = self.root_path + directory
        if self.dataset == 'hmdb51':
            if self.modality == 'rgb':
                directory = "/data/home/awinywang/Data/ft_local/hmdb51/jpegs_256/" + directory.strip().split(' ')[0].split('/')[-1]
            else:
                directory = "/data/home/awinywang/Data/ft_local/hmdb51/tvl1_flow/" + \
                            directory.strip().split(' ')[0].split('/')[-1]
        elif self.dataset == 'ucf101':
            if self.modality == 'rgb':
                directory = "/data/home/awinywang/Data/ft_local/ucf101/jpegs_256/" + directory.strip().split(' ')[0].split('/')[-1]
            else:
                directory = "/data/home/awinywang/Data/ft_local/ucf101/tvl1_flow/" + \
                            directory.strip().split(' ')[0].split('/')[-1]
        else:
            Exception("wrong dataset!")
        rgb_img = cv2.imread(os.path.join(directory, self.image_tmpl.format(idx)))
        if self.dataset == 'ucf101':
            u_img_path = directory + '/frame' + str(idx).zfill(6) + '.jpg'
            v_img_path = directory + '/frame' + str(idx).zfill(6) + '.jpg'
        else:
            u_img_path = os.path.join(directory, self.image_tmpl.format('flow_x', idx))
            v_img_path = os.path.join(directory, self.image_tmpl.format('flow_y', idx))
        x_img = cv2.imread(u_img_path)
        y_img = cv2.imread(v_img_path)
        flow = np.zeros((x_img.shape[0], x_img.shape[1], 2), dtype=np.float32)
        flow[:, :, 0] = x_img[:, :, 0]
        flow[:, :, 1] = y_img[:, :, 0]
        # prob = max(1, np.random.random()*40)
        prob = max(0.01, np.random.random() * 5)
        temporal_wrap_img = warp_flow(rgb_img, (flow / 255 - 1) * prob)
        spatial_warp_img = rgb_img.copy()
        # spatial_warp_img = spatial_warp(rgb_img, spatial_warp_points)
        rgb_img = Image.fromarray(np.uint8(rgb_img))
        temporal_wrap_img = Image.fromarray(np.uint8(temporal_wrap_img))
        spatial_warp_img = Image.fromarray(np.uint8(spatial_warp_img))
        return [rgb_img], [temporal_wrap_img], [spatial_warp_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record, new_length=32):
        """

        :param record: VideoRecord
        :return: list
        """
        index = random.randint(1, max(record.num_frames - new_length * self.stride, 0) + 1)
        return index  # ? return array,because rangint is 0 -> num-1

    def _get_val_indices(self, record):
        if record.num_frames//2 > self.new_length - 1:
            offsets = np.array(record.num_frames//2)
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record, new_length=64):
        if record.num_frames > self.num_segments + new_length * self.stride - 1:
            offsets = np.sort(
                random.sample(range(0, record.num_frames - new_length * self.stride + 1), self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def get_flow(self, record, indices, new_length=16):
        images = list()
        gen_images = list()
        p = int(indices)
        prob = np.random.random() * 10
        for i in range(new_length):
            rgb_img, gen_rgb_img = self._load_gen_image(record.path, p, prob)
            images.extend(rgb_img)
            gen_images.extend(gen_rgb_img)
            if p < record.num_frames - self.stride + 1:
                p += self.stride
            else:
                p = 1
        return images, gen_images, record.label

    def get_flow_2(self, record, indices, new_length=16):
        images = list()
        temporal_wrap_images = list()
        spatial_wrap_images = list()
        p = int(indices)
        for i in range(new_length):
            rgb_img, temporal_wrap_image, spatial_wrap_image = self._load_gen_image_2(record.path, p)
            images.extend(rgb_img)
            spatial_wrap_images.extend(spatial_wrap_image)
            temporal_wrap_images.extend(temporal_wrap_image)
            if p < record.num_frames - self.stride + 1:
                p += self.stride
            else:
                p = 1
        return images, temporal_wrap_images, spatial_wrap_images, record.label

    def get(self, record, indices, new_length=16, is_numpy=False):
        images = list()
        p = int(indices)
        if not self.full_video:
            for i in range(new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames - self.stride + 1:
                    p += self.stride
                else:
                    p = 1
        else:
            p = 1
            if record.num_frames < new_length:
                for i in range(new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames - self.stride + 1:
                        p += self.stride
                    else:
                        p = 1
            else:
                for i in range(record.num_frames):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames - self.stride + 1:
                        p += self.stride
                    else:
                        p = 1
        # images = transform_data(images, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
        if is_numpy:
            frames_up = []
            if self.modality == 'rgb':
                for i, img in enumerate(images):
                    frames_up.append(np.asarray(img))
            elif self.modality == 'flow':
                for i in range(0, len(images), 2):
                    # it is used to combine frame into 2 channels
                    tmp = np.stack([np.asarray(images[i]), np.asarray(images[i + 1])], axis=2)
                    frames_up.append(tmp)
            images = np.stack(frames_up)

            if self.full_video:
                if record.num_frames < self.new_length:
                    images = self.frames_padding(images, self.new_length)
        return images, record.label

    def get_test(self, record, indices):
        '''
        get num_segments data
        '''
        # print(indices)
        all_images = []
        count = 0
        for seg_ind in indices:
            images = []
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                # print(seg_imgs)
                images.append(seg_imgs)
                if p < record.num_frames - self.stride + 1:
                    p += self.stride
                else:
                    p = 1
            all_images.append(images)
            count = count + 1
        process_data = np.asarray(all_images, dtype=np.float32)
        # print(process_data.shape)
        return process_data, record.label

    def get_norm_item(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record, new_length=self.new_length)
            data, label = self.get(record, segment_indices, new_length=self.new_length)
            # data = 2 * (data / 255) - 1
            data = self.transform(data)
            if type(data) == list and len(data) > 1:
                new_data = list()
                for one_sample in data:
                    new_data.append((one_sample))
            else:
                new_data = (data)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
            data, label = self.get(record, segment_indices, new_length=self.new_length)
            # data = 2 * (data / 255) - 1
            data = self.transform(data)
            if type(data) == list and len(data) > 1:
                new_data = list()
                for one_sample in data:
                    new_data.append((one_sample))
            else:
                new_data = (data)
        return new_data, label, index

    # pretrain normalizatio
    def get_moco_items(self, index):
        record = self.video_list[index]  # video name?
        index2 = index
        while index2 == index:
            index2 = random.randint(1, self.__len__()-1)
        record2 = self.video_list[index2]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
        if not self.test_mode:
            negative_segment_indices = self._sample_indices(record2)
        else:
            negative_segment_indices = self._get_test_indices(record2, new_length=self.new_length)
        anchor_data, label = self.get(record, segment_indices)
        postive_data, label = self.get(record, segment_indices)
        # load
        negative_data, label = self.get(record2, negative_segment_indices)
        anchor_data = self.transform(anchor_data)
        postive_data = self.transform(postive_data)
        negative_data = self.transform(negative_data)
        return anchor_data, postive_data, negative_data, label, index

    def get_dsm_items(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
        # postive: same in temporal but spatial
        negative_segment_indices = segment_indices
        # important hyperparameter
        if self.dataset == 'sth_v1':
            index2 = index
            while index2 == index:
                index2 = random.randint(1, self.__len__() - 1)
            record2 = self.video_list[index2]  # video name?
            if not self.test_mode:
                negative_segment_indices = self._sample_indices(record2)
            else:
                negative_segment_indices = self._get_test_indices(record2, new_length=self.new_length)
            negative_data, label = self.get(record2, negative_segment_indices)
        else:
            thresh = 2
            if not self.test_mode:
                negative_segment_indices = self._sample_indices(record)
            else:
                negative_segment_indices = self._get_test_indices(record, new_length=self.new_length)
            if abs(negative_segment_indices - segment_indices) < thresh:
                negative_segment_indices = (negative_segment_indices + record.num_frames // 3) % record.num_frames
            if negative_segment_indices == 0:
                negative_segment_indices += 1
            negative_data, label = self.get(record, negative_segment_indices)
        anchor_data, label = self.get(record, segment_indices)
        postive_data, label = self.get(record, segment_indices)
        # load
        anchor_data = self.transform(anchor_data)
        postive_data = self.transform(postive_data)
        negative_data = self.transform(negative_data)
        return anchor_data, postive_data, negative_data, label, index

    def get_flow_items(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
        thresh = 2
        if not self.test_mode:
            negative_segment_indices = self._sample_indices(record)
        else:
            negative_segment_indices = self._get_test_indices(record, new_length=self.new_length)
        if abs(negative_segment_indices - segment_indices) < thresh:
            negative_segment_indices = (negative_segment_indices + record.num_frames // 3) % record.num_frames
        if negative_segment_indices == 0:
            negative_segment_indices += 1
        datas = []
        anchor_data, temporal_wrap, label = self.get_flow(record, segment_indices)
        # anchor_data, label = self.get(record, segment_indices)
        for i in range(5):
            temp_data = self.transform(anchor_data)
            datas.append(temp_data)
        temporal_wrap = self.transform(temporal_wrap)
        datas.append(temporal_wrap)
        negative_data, label = self.get(record, negative_segment_indices)
        negative_data = self.transform(negative_data)
        datas.append(negative_data)
        return datas, label, index
        # anchor_data, strong_negative_data, label = self.get_flow(record, segment_indices)
        # negative_data, label = self.get(record, negative_segment_indices)
        # anchor_data_origin = self.transform(anchor_data)
        # postive_data = self.transform(anchor_data)
        # negative_data = self.transform(negative_data)
        # strong_negative_data = self.transform(strong_negative_data)
        # return anchor_data_origin, postive_data, negative_data, strong_negative_data, label, index

    def get_flow_items_2(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
        thresh = 2
        if not self.test_mode:
            negative_segment_indices = self._sample_indices(record)
        else:
            negative_segment_indices = self._get_test_indices(record, new_length=self.new_length)
        if abs(negative_segment_indices - segment_indices) < thresh:
            negative_segment_indices = (negative_segment_indices + record.num_frames // 3) % record.num_frames
        if negative_segment_indices == 0:
            negative_segment_indices += 1
        anchor_data, temporal_wrap_data, spatial_wrap_data, label = self.get_flow_2(record, segment_indices, new_length=self.new_length)
        negative_data, label = self.get(record, negative_segment_indices, new_length=self.new_length)
        positive_data, _ = self.get(record, segment_indices, new_length=self.new_length)
        import augment.video_transformations.video_transform_PIL_or_np as video_transform
        from augment.video_transformations.volume_transforms import ClipToTensor
        from torchvision import transforms
        train_transforms = transforms.Compose([
            video_transform.RandomRotation(30),
            video_transform.Resize(256),
            video_transform.RandomCrop(224),
            video_transform.ColorJitter(0.5, 0.5, 0.25, 0.5),
            ClipToTensor(channel_nb=3),
        ])
        anchor = train_transforms(anchor_data)
        # anchor = self.transform(anchor_data)
        positive = self.transform(positive_data)
        negative = self.transform(negative_data)
        temporal_wrap = self.transform(temporal_wrap_data)
        spatial_wrap = self.transform(spatial_wrap_data)
        return [anchor, positive, negative, temporal_wrap, spatial_wrap], label, index

    def get_triplet_items(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_test_indices(record, new_length=self.new_length)
        # postive: same in temporal but spatial
        negative_segment_indices = segment_indices
        # important hyperparameter
        thresh = 2
        if not self.test_mode:
            negative_segment_indices = self._sample_indices(record)
        else:
            negative_segment_indices = self._get_test_indices(record, new_length=self.new_length)
        if abs(negative_segment_indices - segment_indices) < thresh:
            negative_segment_indices = (negative_segment_indices + record.num_frames // 3) % record.num_frames
        if negative_segment_indices == 0:
            negative_segment_indices += 1
        anchor_data, label = self.get(record, segment_indices)
        postive_data, label = self.get(record, segment_indices)
        # load
        negative_data, label = self.get(record, negative_segment_indices)
        anchor_data = self.transform(anchor_data)
        postive_data = self.transform(postive_data)
        negative_data = self.transform(negative_data)
        return anchor_data, postive_data, negative_data, label, index

    def __getitem__(self, index):
        if self.args.status == 'pt':
            if self.args.pt_method == 'dsm_triplet':
                a_1, p_1, n_1, label, index = self.get_triplet_items(index)
                return [a_1, p_1, n_1], label, index
            elif self.args.pt_method == 'dsm':
                anchor_data, postive_data, negative_data, label, index = self.get_dsm_items(index)
                return [anchor_data, postive_data, negative_data], label, index
            elif self.args.pt_method == 'moco':
                anchor_data, postive_data, negative_data, label, index = self.get_moco_items(index)
                return [anchor_data, postive_data, negative_data], label, index
        else:
            data, label, index = self.get_norm_item(index)
            return data, label, index

    def __len__(self):
        return len(self.video_list)
