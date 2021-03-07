from __future__ import print_function, division
import os
import sys
import subprocess


def process(dir_path, dst_dir_path):
  count0 = 0
  for class_name in os.listdir(dir_path):
    count1 = 0
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
      return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
      os.mkdir(dst_class_path)
    for file_name in os.listdir(class_path):
      if file_name[-4:] != '.avi':
        continue
      name, ext = os.path.splitext(file_name)
      dst_directory_path = os.path.join(dst_class_path, name)

      video_file_path = os.path.join(class_path, file_name)
      try:
        if os.path.exists(dst_directory_path):
          if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
            subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
            print('remove {}'.format(dst_directory_path))
            os.mkdir(dst_directory_path)
          else:
            continue
        else:
          os.mkdir(dst_directory_path)
      except:
        print(dst_directory_path)
        continue
      cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
      # print(cmd)
      subprocess.call(cmd, shell=True)
      count1 += 1
      print("{}/{} classes: {}/{} videos finished".format(count0, len(os.listdir(dir_path)), count1, len(os.listdir(class_path))))
      # print('\n')
    count0 += 1


if __name__=="__main__":
  # /data1/DataSet/hmdb51_sta_new
  # /data1/DataSet/hmdb51_sta_frames
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  process(dir_path, dst_dir_path)
