import numpy as np
import lintel

'''
    video decode on the fly based on https://github.com/dukebw/lintel
'''

def _sample_frame_sequence_to_4darray(video, dataset, should_random_seek, fps_cap):
    """Called to extract a frame sequence `dataset.num_frames` long, sampled
    uniformly from inside `video`, to a 4D numpy array.
.
    Args:
        video: Encoded video.
        dataset: Dataset meta-info, e.g., width and height.
        should_random_seek: If set to `True`, then `lintel.loadvid` will start
            decoding from a uniformly random seek point in the video (with
            enough space to decode the requested number of frames).
            The seek distance will be returned, so that if the label of the
            data depends on the timestamp, then the label can be dynamically
            set.
        fps_cap: The _maximum_ framerate that will be captured from the video.
            Excess frames will be dropped, i.e., if `fps_cap` is 30 for a video
            with a 60 fps framerate, every other frame will be dropped.
    Returns:
        A tuple (frames, seek_distance) where `frames` is a 4-D numpy array
        loaded from the byte array returned by `lintel.loadvid`, and
        `seek_distance` is the number of seconds into `video` that decoding
        started from.
    Note that the random seeking can be turned off.
    Use _sample_frame_sequence_to_4darray in your PyTorch Dataset object, which
    subclasses torch.utils.data.Dataset. Call _sample_frame_sequence_to_4darray
    in __getitem__. This means that for every minibatch, for each example, a
    random keyframe in the video is seeked to and num_frames frames are decoded
    from there. num_frames would normally tend to be small (if you were going
    to use them as input to a 3D ConvNet or optical flow algorithm), e.g., 32
    frames.
    """
    video, seek_distance = lintel.loadvid(
        video,
        should_random_seek=should_random_seek,
        width=dataset.width,
        height=dataset.height,
        num_frames=dataset.num_frames,
        fps_cap=fps_cap)
    video = np.frombuffer(video, dtype=np.uint8)
    video = np.reshape(
        video, newshape=(dataset.num_frames, dataset.height, dataset.width, 3))

    return video, seek_distance


def _load_frame_nums_to_4darray(video, dataset, frame_nums):
    """Decodes a specific set of frames from `video` to a 4D numpy array.
    Args:
        video: Encoded video.
        dataset: Dataset meta-info, e.g., width and height.
        frame_nums: Indices of specific frame indices to decode, e.g.,
            [1, 10, 30, 35] will return four frames: the first, 10th, 30th and
            35 frames in `video`. Indices must be in strictly increasing order.
    Returns:
        A numpy array, loaded from the byte array returned by
        `lintel.loadvid_frame_nums`, containing the specified frames, decoded.
    """
    decoded_frames = lintel.loadvid_frame_nums(video,
                                               frame_nums=frame_nums,
                                               width=dataset.width,
                                               height=dataset.height)
    decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
    decoded_frames = np.reshape(
        decoded_frames,
        newshape=(dataset.num_frames, dataset.height, dataset.width, 3))

    return decoded_frames

def _sample_action_frame_sequence_to_4darray(video, num_frames, should_random_seek, fps_cap):
    """Called to extract a frame sequence `dataset.num_frames` long, sampled
    uniformly from inside `video`, to a 4D numpy array.
.
    Args:
        video: Encoded video.
        should_random_seek: If set to `True`, then `lintel.loadvid` will start
            decoding from a uniformly random seek point in the video (with
            enough space to decode the requested number of frames).
            The seek distance will be returned, so that if the label of the
            data depends on the timestamp, then the label can be dynamically
            set.
        fps_cap: The _maximum_ framerate that will be captured from the video.
            Excess frames will be dropped, i.e., if `fps_cap` is 30 for a video
            with a 60 fps framerate, every other frame will be dropped.
    Returns:
        A tuple (frames, seek_distance) where `frames` is a 4-D numpy array
        loaded from the byte array returned by `lintel.loadvid`, and
        `seek_distance` is the number of seconds into `video` that decoding
        started from.
    Note that the random seeking can be turned off.
    Use _sample_frame_sequence_to_4darray in your PyTorch Dataset object, which
    subclasses torch.utils.data.Dataset. Call _sample_frame_sequence_to_4darray
    in __getitem__. This means that for every minibatch, for each example, a
    random keyframe in the video is seeked to and num_frames frames are decoded
    from there. num_frames would normally tend to be small (if you were going
    to use them as input to a 3D ConvNet or optical flow algorithm), e.g., 32
    frames.
    """
    #this function will auto loop is suppress video length

    video, width, height, seek_distance = lintel.loadvid(
        video,
        should_random_seek=should_random_seek,
        num_frames=num_frames)
    video = np.frombuffer(video, dtype=np.uint8)
    video = np.reshape(video, newshape=(num_frames, height, width, 3))

    return video, seek_distance

def _load_action_frame_nums_to_4darray(video, frame_nums, width, height):
    """Decodes a specific set of frames from `video` to a 4D numpy array.
    Args:
        video: Encoded video.
        dataset: Dataset meta-info, e.g., width and height.
        frame_nums: Indices of specific frame indices to decode, e.g.,
            [1, 10, 30, 35] will return four frames: the first, 10th, 30th and
            35 frames in `video`. Indices must be in strictly increasing order.
    Returns:
        A numpy array, loaded from the byte array returned by
        `lintel.loadvid_frame_nums`, containing the specified frames, decoded.
    """
    decoded_frames = lintel.loadvid_frame_nums(video,
                                               frame_nums=frame_nums,
                                               width=width,
                                               height=height)
    decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
    decoded_frames = np.reshape(
        decoded_frames,
        newshape=(len(frame_nums), height, width, 3))

    return decoded_frames

# import torchvision
# # read dataset from pytorch API
# import classy_vision
#
# def dataset_config(args):
#     if args.dataset == 'kinetics':
#         dataset = classy_vision.dataset.Kinetics400Dataset(split: 'train', batchsize_per_replica: 8,
#         shuffle: True, transform: None, num_samples: None,
#         frames_per_clip: 8, video_width: 256, video_height: 340,
#         video_min_dimension: 256, audio_samples: 0, audio_channels: 0,
#         step_between_clips: 4, frame_rate:0,
#         clips_per_video: int, video_dir: str,
#         extensions: 'avi', metadata_filepath: str)
#
#     elif args.dataset == 'hmdb51':
#         torchvision.datasets.HMDB51(root, annotation_path, frames_per_clip,
#                                     step_between_clips=1, frame_rate=None, fold=1,
#                                     train=True, transform=None, _precomputed_metadata=None,
#                                     num_workers=1, _video_width=0, _video_height=0,
#                                     _video_min_dimension=0, _audio_samples=0)
#     elif args.dataset == 'ucf101':
#         torchvision.datasets.UCF101(root, annotation_path, frames_per_clip,
#                                     step_between_clips=1, frame_rate=None, fold=1,
#                                     train=True, transform=None, _precomputed_metadata=None,
#                                     num_workers=1, _video_width=0, _video_height=0,
#                                     _video_min_dimension=0, _audio_samples=0)
#     else:
#         Exception("wrong dataset")