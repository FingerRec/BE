# Easy data augmentation techniques for video
# Jason Wei and Kai Zou

import torch.nn as nn
import random
from random import shuffle
#random.seed(1)
# cleaning up text
import re



########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(videos, p):
    # obviously, if there's only one word, don't delete it
    b, c, t, h, w = videos.size()
    if t == 1:
        return videos

    # randomly delete words with probability p, padding or loop?
    '''
    # method 1
    new_videos = videos.copy()
    for i in range(t):
        r = random.uniform(0, 1)
        if r <= p:
            new_videos[:, :, i, :, :] = 0
    '''
    # method 2 loop
    new_videos = videos
    count = 0
    for i in range(t):
        r = random.uniform(0, 1)
        if r <= p:
            continue
        else:
            new_videos[:, :, count, :, :] = videos[:, :, i, :, :,]
        count += 1
    for i in range(t - count):
        new_videos[:, :, i, :, :] = videos[:, :, i, :, :]
    # if you end up deleting all words, just return a random word
    if new_videos.size()[2] == 0:
        rand_int = random.randint(0, t - 1)
        return [videos[:, :, rand_int, :, :]]

    return new_videos


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(videos, n):
    new_videos = videos
    for _ in range(n):
        new_videos = swap_word(new_videos)
    return new_videos


def swap_word(new_videos):
    b, c, t, h, w = new_videos.size()
    random_idx_1 = random.randint(0, t - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, t - 1)
        counter += 1
        if counter > 3:
            return new_videos
    new_videos[:, :, random_idx_1, :, :], new_videos[:, :, random_idx_2, :, :] \
        = new_videos[:, :, random_idx_2, :, :], new_videos[:, :, random_idx_1, :, :]
    return new_videos


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(videos, n):
    new_videos = videos
    for _ in range(n):
        new_videos = add_picture(new_videos)
    return new_videos


def add_picture(videos):
    b, c, t, h, w = videos.size()
    new_videos = videos
    random_idx = random.randint(0, t - 1)
    random_idx2 = random.randint(0, t - 1)
    # this is from the same sample, may be need modify
    new_videos[:, :, random_idx+1:, :, :] = videos[:, :, random_idx:t-1, :, :]
    new_videos[:, :, random_idx, :, :] = videos[:, :, random_idx2, :, :]
    return new_videos

########################################################################
# main data augmentation function
########################################################################


class VideoEda(nn.Module):
    def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
        super(VideoEda, self).__init__()
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd
        self.num_aug = num_aug

    def eda(self, inp):
        b, c, t, h, w = inp.size()

        augmented_sentences = []
        num_new_per_technique = int(self.num_aug / 4) + 1
        n_sr = max(1, int(self.alpha_sr * t))
        n_ri = max(1, int(self.alpha_ri * t))
        n_rs = max(1, int(self.alpha_rs * t))

        # # sr synonym replacement
        # for _ in range(num_new_per_technique):
        #     inp = synonym_replacement(inp, n_sr)

        # ri random insertion
        for _ in range(num_new_per_technique):
            inp = random_insertion(inp, n_ri)

        # rs random swap
        for _ in range(num_new_per_technique):
            inp = random_swap(inp, n_rs)

        # # rd random delte
        # for _ in range(num_new_per_technique):
        #     inp = random_deletion(inp, self.p_rd)

        return inp

    def forward(self, x):
        return self.eda(x)