import lintel
import numpy as np
import cv2
import skvideo.io

def video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        #print("could not open: ", video_path)
        return -1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    return length, width, height

v_path = "/data1/DataSet/Kinetics/compress/val_256/bee_keeping/p3_wp0Cq6Lo.mp4"
video_frames_num, width, height = video_frame_count(v_path)

# video, width, height, seek_index = lintel.loadvid(open(v_path, 'rb').read(), should_random_seek=False)
# video = np.reshape(np.frombuffer(video, dtype=np.uint8), (-1, height, width, 3))
# num_frames = video.shape
# print(video_frames_num, num_frames)
#
# videodata = skvideo.io.vread(v_path, inputdict={'-r': '4'})
# print(videodata.shape)
print(video_frames_num)
f = open(v_path, 'rb')
video = f.read()
f.close()

# ffmpeg count 比 cv2少1帧
frame_nums = [0,201,280, 295, 296]
decoded_frames = lintel.loadvid_frame_nums(video,
                                               frame_nums=frame_nums,
                                               width=width,
                                               height=height)
decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
decoded_frames = np.reshape(
    decoded_frames,
    newshape=(len(frame_nums), height, width, 3))

print(np.shape(decoded_frames)[0])

# pytorch vision里的Kinetics dataset可以简单改写，可以得到音频，但是不能限定开始的index
# self.video_clips = VideoClips(
#             video_list,
#             frames_per_clip,
#             step_between_clips,
#             frame_rate,
#             _precomputed_metadata,
#             num_workers=num_workers,
#             _video_width=_video_width,
#             _video_height=_video_height,
#             _video_min_dimension=_video_min_dimension,
#             _audio_samples=_audio_samples,
#             _audio_channels=_audio_channels,
#         )