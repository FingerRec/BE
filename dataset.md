# Dataset Prepare
For Kinetics, we decode on the fly, each row in the txt file include:
> video_path class

And we load the videos directly, please place the training set in SSD for fast IO.

Prepare dataset (UCF101/diving48/sth/hmdb51/actor-hmdb51), and each row of txt is as below:

> video_path class frames_num

These datasets saved in frames. We offer list for all these datasets in [Google Driver](https://drive.google.com/drive/folders/1ndq0rdxEvubBrbXny8RuGCTETXU2hr1N?usp=sharing).

## Kinetics
As some Youtube Link is lost, we use a copy of kinetics-400 from [Non-local](https://github.com/facebookresearch/video-nonlocal-net), the training set is 234643 videos now and the val set is 19761 now. 
All the videos in mpeg/avi format.

## UCF101/HMDB51
These two video datasets contain three train/test splits. 
Down UCF101 from [crcv](https://www.crcv.ucf.edu/data/UCF101.php)  and HMDB51 from [serre-la](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

## Diving48
(1). Download divin48 from [ucsd](http://www.svcl.ucsd.edu/projects/resound/dataset.html)

(2). Generated frames using script __src/Contrastive/utils/data_process/gen_diving48_frames.py__

(3). Generated lists using script __src/Contrastive/utils/data_process/gen_diving48_lists.py__

## Sth-v1

## HMDB51-STA/Actor-HMDB51
The HMDB51-STA is removing huge cmaera motion from [HMDB-STA](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
We provide our generated Actor-HMDB51 in [google_driver]().

### Semi-supervised Subdataset
We also provide manuscript to get the sub-set of kinetics-400. Please refer to Contrastive/utils/data_provess/semi_data_split.py for details.
