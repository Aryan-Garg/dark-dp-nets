#!/usr/bin/env python3

import os
import cv2
import numpy as np
from tqdm.auto import tqdm

base_dir = "/media/girish/Elements/disp_pixel4_BA/"

overall_mean = 0.
overall_std = 0.
total_n = 0

video_mean = {}
video_std = {}

overall_max_val = 0.
overall_min_val = 100000.

pbar = tqdm(os.listdir(base_dir))
for video in (pbar):
    pbar.set_description(f"Processing {video}")
    if not os.path.isdir(os.path.join(base_dir, video)):
        continue
    video_dir = os.path.join(base_dir, video)
    video_mean[video] = [0., 0.]
    # print(f"Processing {video_dir}")
    frames = os.listdir(video_dir)
    frames = [os.path.join(video_dir, frame) for frame in frames if frame.endswith(".pfm")]

    mean_std = []
    for frame in frames:
        img = cv2.imread(frame, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        mean = np.mean(img)
        std = np.std(img)
        # print(img.shape)
        # mean = np.mean(img, axis=(0, 1))
        # std = np.std(img, axis=(0, 1))
        # mean_std.append((mean, std))
        overall_mean += mean
        overall_std += std

        overall_max_val = max(overall_max_val, np.amax(img))
        overall_min_val = min(overall_min_val, np.amin(img))
    total_n += len(frames)

print("Overall mean: ", overall_mean / total_n)
print("Overall std: ", overall_std / total_n)
print("Overall max val: ", overall_max_val)
print("Overall min val: ", overall_min_val)

# print("Video mean: ", video_mean)
# print("Video std: ", video_std)