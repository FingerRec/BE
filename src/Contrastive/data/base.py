import torch
import cv2


# from config import opt
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    # return torch.from_numpy(pic)
    return torch.from_numpy(pic.transpose([3, 0, 1, 2])).type(torch.FloatTensor)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1]) - 1

    @property
    def label(self):
        return int(self._data[2])


def video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        #print("could not open: ", video_path)
        return -1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    return length, width, height

#
# if __name__ == '__main__':
#     print("")