#!/usr/bin/env python3
import os

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

# import tensorflow as tf
from torchvision.transforms import transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

from torchvision.utils import make_grid

class depthEstimationDataset(Dataset):
    def __init__(self, files, mode="train", input_type: str = "rgb_dp", transform=A.Compose([ToTensorV2()])):
        self.files = files
        self.mode = mode

        self.input_type = input_type
        
        self.height = 224
        self.width = 224

        self.transform =  A.Compose([
            A.RandomCrop(width=self.width, height=self.height),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()])

        self.rgb_normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                   std=[0.229, 0.224, 0.225]) # use image net mean and std
        
        self.test_transform = A.Compose([
            A.Resize(width=self.width, height=self.height),
            ToTensorV2()])


    def __len__(self):
        with open(self.files, "r") as f:
            return len(f.readlines())


    def __getitem__(self, index):
        with open(self.files, "r") as f:
            lines = f.readlines()
            rgb_file, dp_left_file, dp_right_file, disp_file = lines[index][:-1].split(' ')

            rgb = cv2.imread(rgb_file)
            rgb = (cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) / 255.).astype(np.float32)

            # rgb = cv2.resize(rgb, (self.width, self.height))
            
            left_pd_img = np.array(Image.open(dp_left_file).rotate(
                270, expand=True
            ).resize((600, 800)), dtype=np.float32)[:,:,0:1]  # match RGB and disp shape first
           
            right_pd_img = np.array(Image.open(dp_right_file).rotate(
                270, expand=True
            ).resize((600, 800)), dtype=np.float32)[:,:,0:1] # match RGB and disp shape first
            
            
            # Undo tonemap and normalize 16 bit DP data
            left_pd_img = (left_pd_img)**2 / ((2**16-1)*1.0)
            right_pd_img = (right_pd_img)**2 / ((2**16-1)*1.0)
            # print("Max left_pd_img:", np.amax(left_pd_img), "Min left_pd_img:", np.amin(left_pd_img))
            # print("Max right_pd_img:", np.amax(right_pd_img), "Min right_pd_img:", np.amin(right_pd_img))

            disp = cv2.imread(disp_file, cv2.IMREAD_ANYDEPTH).astype(np.float32) 

            # NOTE:Remove standardization of disp data; NO BN used in decoder either.
            # disp = (disp - 18.458856649276722) / 7.963991303429185 # standardization using disp data statistics : (mu, std)
            # print("Max disp:", np.amax(disp), "Min disp:", np.amin(disp))
            disp = np.expand_dims(disp, axis=2)
            # disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            
            # print(np.amax(depth), np.amin(depth))   
            # plt.imsave(f'./depth_{index}.png', depth, cmap='inferno')
            
            if "rgb_dp" == self.input_type: # NOTE: Proposed Solution
                # print(rgb.shape, rgb.dtype, np.amax(rgb), np.amin(rgb))
                dp = np.concatenate((left_pd_img, right_pd_img), axis=2)
                if self.mode == "train":
                    X = self.transform(image=np.concatenate((rgb, dp, disp), axis=2))["image"]
                else:
                    X = self.test_transform(image=np.concatenate((rgb, dp, disp), axis=2))["image"]
                X[:3, ...] = self.rgb_normalize_transform(X[:3, ...]) # note this is a torchvision transform
            elif "rgb" == self.input_type:
                if self.mode == "train":
                    X = self.transform(image=np.concatenate(rgb, disp), axis=2)["image"]
                else:
                    X = self.test_transform(image=np.concatenate(rgb, disp), axis=2)["image"]
                X[:3, ...] = self.rgb_normalize_transform(X[:3, ...]) # note this is a torchvision transform
            elif "dp" == self.input_type: 
                if self.mode == "train":
                    X = self.transform(image=np.concatenate((left_pd_img, right_pd_img, disp), axis=2))["image"]
                else:
                    X = self.test_transform(image=np.concatenate((left_pd_img, right_pd_img, disp), axis=2))["image"]
            else:
                raise ValueError("Invalid input type. Please provide either 'rgb', 'dp' or 'rgb_dp' as input type.")
            
            # print("X:", X.shape)
            
            return X
        

class BroDataset(Dataset):
    def __init__(self, txt_files, mode="train", rgb_input=False):
        self.rgb_input = rgb_input
        self.unrect_datapath = '/data2/raghav/datasets/Pixel4_3DP/unrectified'
        self.datapath = '/girish/media/Elements/'
        self.mode = mode

        # model_type = "DPT_Large"   
        # self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        # self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        # if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        #     self.midas_transform = self.midas_transforms.dpt_transform
        # else:
        #     self.midas_transform = self.midas_transforms.small_transform

        if mode == "val":
            with open(os.path.join(txt_files, 'val_files.txt'), "r") as f:
                self.filenames = f.readlines()
            # print("Validation dataset length: ", len(self.filenames))

        elif mode == "train":
            with open(os.path.join(txt_files, 'train_files.txt'), "r") as f:
                self.filenames = f.readlines()
            # print("Train dataset length: ", len(self.filenames))

        elif mode == 'test':
            with open(os.path.join(txt_files, 'test_files.txt'), "r") as f:
                self.filenames = f.readlines()
            # print("Test dataset length: ", len(self.filenames))
        self.height = 224
        self.width = 224

        self.transform = T.Compose([
            T.Resize((self.height, self.width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def __len__(self) -> int:
        return len(self.filenames)


    def __getitem__(self, index):
        rgb_file, dp_left_file, dp_right_file, dpt_file = self.filenames[index][:-1].split(';')
        
        center_img = cv2.imread(os.path.join(
            self.datapath, "B", "Video_data", rgb_file
        ))
        center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
        input_batch = self.midas_transform(center_img.copy())

        left_pd_img = np.array(Image.open(os.path.join(
            self.unrect_datapath, "B", "dp_data", dp_left_file
        )).rotate(
            270, expand=True
        ).resize((self.width, self.height)), dtype=np.float32)[:,:,0:1] 

        right_pd_img = np.array(Image.open(os.path.join(
            self.unrect_datapath, "B", "dp_data", dp_right_file
        )).rotate(
            270, expand=True
        ).resize((self.width, self.height)), dtype=np.float32)[:,:,0:1]

        # Undo tonemap and normalize 16 bit DP data
        left_pd_img = (left_pd_img)**2 / ((2**16-1)*1.0)
        right_pd_img = (right_pd_img)**2 / ((2**16-1)*1.0)

        # normalize dp data
        left_pd_img = (left_pd_img - np.amin(left_pd_img)) / (np.amax(left_pd_img) - np.amin(left_pd_img))
        right_pd_img = (right_pd_img - np.amin(right_pd_img)) / (np.amax(right_pd_img) - np.amin(right_pd_img))     
        
        disp = cv2.imread(os.path.join("/data2/aryan/lfvr/disparity_maps/disp_pixel4_BA", 
                                    dpt_file.split('.')[0] + '_disp.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32)
        # print(disp.shape, np.amax(disp), np.amin(disp))
        disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_CUBIC) / 255.
        
        # with torch.no_grad():
        #     prediction = self.midas(input_batch)
        #     prediction = torch.nn.functional.interpolate(
        #         prediction.unsqueeze(1),
        #         size=(self.height, self.width),
        #         mode="bicubic",
        #         align_corners=False,
        #     ).squeeze()

        # depth = (1. / prediction).cpu().numpy().astype(np.float32)
        # depth = (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))
        # print(depth.shape, np.amax(depth), np.amin(depth))

        # add channel dimension to depth map
        depth = torch.from_numpy(np.expand_dims(depth, axis=2)).permute(2,0,1)
        dp_input = torch.from_numpy(np.concatenate((left_pd_img, right_pd_img), axis=2)).permute(2,0,1)

        if self.rgb_input:
            norm_center_img = self.transform(Image.fromarray(np.uint8(center_img)))
            dp_input = torch.cat([dp_input, norm_center_img], dim=0)
        # print(dp_input.shape)
        sample = {'dp_input': dp_input, 'disp': disp}

        return sample


def denormalize3d(x, device='cpu'):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    #print(x.device, mean.device, std.device)
    return x * std + mean


if __name__ == '__main__':
    dataset = BroDataset("./files/train_files.txt", )
    dLoader = DataLoader(dataset=dataset, batch_size=1)

    for i, X in enumerate(dLoader):
        print(i)
        # rgb = denormalize3d(sample['dp_input'][:,2:,:,:]).squeeze().permute(1,2,0).numpy()
        # print(rgb.shape, np.amax(rgb), np.amin(rgb))
        # plt.imsave(f'{i+1}_right_pd.png', sample['dp_input'][0,1,:,:].numpy()*255)
        # plt.imsave(f'{i+1}_left_pd.png', sample['dp_input'][0,0,:,:].numpy()*255)
        # plt.imsave(f'{i+1}_depth.png', sample['depth'][0,0,:,:].numpy()*255)
        # plt.imsave(f"{i+1}_rgb.png", rgb)
        break
        