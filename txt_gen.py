#!/usr/bin/env python

import os
import numpy as np
import cv2
from tqdm.auto import tqdm

rgb_path = "/media/girish/Elements/datasets/B/Video_data/"
dp_path = "/media/girish/Elements/datasets/dp_dataset/unrectified/B/dp_data/"
disp_path = "/media/girish/Elements/disp_pixel4_BA/"

def gen_all_files():
    skip_frames = 10
    with open("skip10_all_files.txt", "w+") as f:
        for dir in tqdm(os.listdir(rgb_path)):
            i = 0
            for file in os.listdir(os.path.join(rgb_path, dir)):
                if not os.path.exists(os.path.join(dp_path, dir) + "/" + file[:-4] + "_left.jpg"):
                    # print(f"Missing {os.path.join(dp_path, dir) + '/' + file[:-4] + '_left.jpg'}")
                    continue
                if i % skip_frames == 0:
                    dp_right = os.path.join(dp_path, dir) + "/" + file[:-4] + "_right.jpg"
                    dp_left = os.path.join(dp_path, dir) + "/"  + file[:-4] + "_left.jpg"
                    disp = os.path.join(disp_path, dir) + "/"  + file[:-4] + "_disp.pfm"
                    f.write(f"{os.path.join(rgb_path, dir, file)} {dp_left} {dp_right} {disp}\n")
                i += 1

def train_test_split():
    with open("skip10_all_files.txt", "r") as f:
        lines = f.readlines()
        # np.random.shuffle(lines)
        n = len(lines)
        train = lines[:int(0.8*n)]
        val = lines[int(0.8*n):]
        with open("skip10_train_files.txt", "w+") as f:
            f.writelines(train)
        with open("skip10_val_files.txt", "w+") as f:
            f.writelines(val)

gen_all_files()
train_test_split()