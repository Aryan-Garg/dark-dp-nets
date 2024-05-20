# ICCP 2024: Stereo-Knowledge Distillation from dpMV to Dual Pixels for Light Field Video Reconstruction   


> Aryan Garg, Raghav Mallampali, Akshat Joshi, Shrisudhan Govindarajan, Kaushik Mitra     


## Dark Tiny dp-Disparity Estimators 

> Using practical/deployable neural networks for disparity estimation with highest fidelity.

## Usage:

`conda create env > <envfilename-in-repo>`

To train:

Generate text files listing paths of rgb, dp_l, dp_r and gt disp and place under dir: `files` then run:

`./train.py`

To eval:
