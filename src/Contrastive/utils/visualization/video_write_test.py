import cv2
import skvideo.io
import numpy as np
import os
import augment.video_transformations.video_transform_PIL_or_np as video_transform
from torchvision import transforms
import skimage.transform
import random
from PIL import Image
import torchvision


def read_video(video):
    cap = cv2.VideoCapture(video)
    frames = list()
    while True:
        ret, frame = cap.read()
        if type(frame) is type(None):
            break
        else:
            frames.append(frame)
    return frames


def write_video(name, frames):
    # fshape = frames[0].shape
    # fheight = fshape[0]
    # fwidth = fshape[1]
    # writer = cv2.VideoWriter(name,
    #                          cv2.VideoWriter_fourcc(*"MJPG"), 30, (fheight, fwidth))
    # for i in range(len(frames)):
    #     writer.write(frames[i])
    # writer.release()
    writer = skvideo.io.FFmpegWriter(name,
                                    outputdict={'-b': '300000000'})
    for frame in frames:
        frame = np.array(frame)
        writer.writeFrame(frame)
    writer.close()
    return 1


if __name__ == '__main__':
    # video = "../experiments/test.mp4"
    # frames = read_video(video)

    # # aug = video_transform.RandomRotation(10),
    # # video_transform.STA_RandomRotation(10),
    # # video_transform.Each_RandomRotation(10),
    # # train_transforms = transforms.Compose([video_transform.Resize(128),
    # #     video_transform.RandomCrop(112), aug])
    # frames = np.array(frames)
    # prefix = "random_rotation"
    # angle = random.uniform(-45, 45)
    # rotated = [skimage.transform.rotate(img, angle) for img in frames]
    # name = '../experiments/gen_videos/test_{}.avi'.format(prefix)
    # write_video(name, rotated)
    # prefix = "STA_rotation"
    # bsz = len(frames)
    # angles = [(i + 1) / (bsz + 1) * angle for i in range(bsz)]
    # rotated = [skimage.transform.rotate(img, angles[i]) for i, img in enumerate(frames)]
    # name = '../experiments/gen_videos/test_{}.avi'.format(prefix)
    # write_video(name, rotated)
    # prefix = "each_random_rotation"
    # angles = [random.uniform(-45, 45) for i in range(bsz)]
    # rotated = [skimage.transform.rotate(img, angles[i]) for i, img in enumerate(frames)]
    # name = '../experiments/gen_videos/test_{}.avi'.format(prefix)
    # write_video(name, rotated)
    # # for i in range(10):
    # #     seqs = np.load("{}_{}.npy".format("../experiments/augmentation/{}".format(prefix), i))
    # #     seqs = (seqs+1)/2*255
    # #     out_dir = "../experiments/gen_videos/{}".format(i)
    # #     if not os.path.exists(out_dir):
    # #         os.makedirs(out_dir)
    # #     for j in range(16):
    # #         cv2.imwrite("{}/{}_{}.jpg".format(out_dir, prefix, j), seqs[j])
    #     # name = '../experiments/gen_videos/test_{}.mp4'.format(i)
    #     # write_video(name, seqs)
    # video = "../experiments/test.mp4"
    # frames = read_video(video)
    # images = []
    # frames = np.array(frames)
    # for i in range(len(frames)):
    #     img = Image.fromarray(np.uint8(frames[i]))
    #     images.append(img)
    # # Create img transform function sequence
    # img_transforms = []
    # brightness = 0.5
    # contrast = 0.5
    # saturation = 0.5
    # hue = 0.2
    # if brightness is not None:
    #     img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    # if saturation is not None:
    #     img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    # if hue is not None:
    #     img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    # if contrast is not None:
    #     img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    # random.shuffle(img_transforms)
    #
    # # Apply to all images
    # jittered_clip = []
    # for img in images:
    #     for func in img_transforms:
    #         jittered_img = func(img)
    #     jittered_clip.append(jittered_img)
    # name = '../experiments/gen_videos/test_{}.avi'.format('jitter')
    # write_video(name, jittered_clip)
    video = "../experiments/test.mp4"
    frames = read_video(video)
    images = []
    frames = np.array(frames)
    for i in range(len(frames)):
        img = Image.fromarray(np.uint8(frames[i]))
        images.append(img)
    # Create img transform function sequence
    brightness = 0.5
    contrast = 0.5
    saturation = 0.5
    hue = 0.2
    # img_transforms = []
    # img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    # img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    # img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    # img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    # random.shuffle(img_transforms)
    #
    # # Apply to all images
    # jittered_clip = []
    # for img in images:
    #     for func in img_transforms:
    #         jittered_img = func(img)
    #     jittered_clip.append(jittered_img)
    # name = '../experiments/gen_videos/test_{}.avi'.format('sta_jitter')
    # write_video(name, jittered_clip)
    # Apply to all images
    jittered_clip = []
    for i, img in enumerate(images):
        t_brightness = (i+1)/(len(images)+1) * brightness
        t_contrast = (i + 1) / (len(images) + 1) * contrast
        t_saturation = (i + 1) / (len(images) + 1) * saturation
        t_hue = (i + 1) / (len(images) + 1) * hue
        img_transforms = []
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, t_brightness))
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, t_saturation))
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, t_hue))
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, t_contrast))
        random.shuffle(img_transforms)
        for func in img_transforms:
            jittered_img = func(img)
        jittered_clip.append(jittered_img)
    name = '../experiments/gen_videos/test_{}.avi'.format('sta_jitter')
    write_video(name, jittered_clip)