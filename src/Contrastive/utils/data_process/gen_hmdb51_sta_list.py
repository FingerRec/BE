import json


out = []
count = 0
front = ''
with open('../datasets/lists/hmdb51/lists/Diving48_V2_train.json', 'r') as f:
    lines = json.load(f)
    for line in lines:
        # line = line.strip()
        count += 1
        new_line = front + str(line['vid_name']) + ' ' + str(line['end_frame']) + ' ' + str(line['label']) + '\n'
        out.append(new_line)
        if count % 100 == 0 and count != 0:
            print(count)

with open('../datasets/lists/diving48/diving48_v2_train_no_front.txt', 'a') as f:
    f.writelines(out)