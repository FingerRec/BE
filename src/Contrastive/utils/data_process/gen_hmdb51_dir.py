from __future__ import print_function, division
import os
import sys
import subprocess


def process(dir_path, dst_dir_path):
  count0 = 0
  for class_name in os.listdir(dir_path):
    count1 = 0
    class_path = os.path.join(dir_path, class_name)
    cmd = 'mv {}/* {}'.format(class_path, dst_dir_path)
    subprocess.call(cmd, shell=True)
    count1 += 1
    count0 += 1


if __name__=="__main__":
  # /data1/DataSet/hmdb51_sta_frames
  # /data1/DataSet/hmdb51_sta_frames2
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  process(dir_path, dst_dir_path)
