import os

def data_split(path, new_file_path, radio):
    video_list = [x for x in open(path)]
    # if not os.path.exists(new_file_path):
    count = 0
    new_list = []
    for line in video_list:
        count += 1
        if count % radio == 0:
            new_list.append(line)
    with open(new_file_path, 'w') as f:
        for item in new_list:
            f.write("%s" % item)

if __name__ == '__main__':
    radio = 10
    path = "../datasets/lists/kinetics-400/ssd_kinetics_video_trainlist.txt"
    new_file_path = "../datasets/lists/kinetics-400/ssd_kinetics_video_trainlist_{}of{}.txt".format(1, radio)
    data_split(path, new_file_path, radio)