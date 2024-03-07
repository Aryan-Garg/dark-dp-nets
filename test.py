#!/usr/bin/env python3

import os
import numpy as np
import cv2

import torch
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from dataset import BroDataset
from loss import SmoothLoss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return {"abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse, 
            "log_rmse": rmse_log, "del_1": a1, "del_2": a2, "del_3": a3}


def getModel(input_type):
    # NOTE - To Try: 
    # 1. resnet50 (inspired from DeepLens)

    if input_type == 'rgb_dp':
        in_chans = 5
    elif input_type == 'rgb':
        in_chans = 3
    elif input_type == 'dp':
        in_chans = 2

    model = smp.UnetPlusPlus(
        encoder_name="timm-mobilenetv3_large_100", # choose deployable models 
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        decoder_use_batchnorm=False,    # don't use batchnorm for decoder (results in droplet artifacts for ResNet50 backbone)
        decoder_attention_type=None,    # don't use attention layers in decoder. Options: [None, scse] # NOTE: Experiment?
        in_channels=in_chans,           # model input channels (2 for dp channels + 3 for RGB)
        classes=1,                      # model output channels (disp is 1 channel)
        activation=None,                # regress the disparity directly; no last layer activation 
    )

    # summary(model.to(device), (in_chans, 224, 224))
    
    return model


def test(experiment_name, model, rgb_input):
    os.makedirs(f'./save_{experiment_name}', exist_ok=True)

    txt_files="/data2/aryan/lfvr/train_inputs/dummy_run"

    ckpts = os.listdir(f'./checkpoints/{experiment_name}')
    ckpt = torch.load(f'./checkpoints/{experiment_name}/{ckpts[-1]}', map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    test_loader = DataLoader(BroDataset(txt_files, mode='test', rgb_input=rgb_input), batch_size=6, shuffle=True)

    model.to(device)
    for epoch in range(1):
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, sample in pbar:
            dp_input, disp = sample['dp_input'], sample['disp']
            dp_input, disp = dp_input.to(device), disp.to(device)
            
            pred = model(dp_input).cpu().detach()
            metrics = compute_errors(disp.cpu().numpy(), pred.numpy())
            print(metrics)
            for i in range(pred.shape[0]):
                plt.imsave(f'./save_{experiment_name}/{i+1}_pred.png', pred[i,0,:,:].numpy()*255)
                plt.imsave(f'./save_{experiment_name}/{i+1}_dp_left.png', dp_input[i,0,:,:].cpu().numpy()*255)
                plt.imsave(f"./save_{experiment_name}/{i+1}_gt.png", disp[i,0,:,:].cpu().numpy()*255)
            # loss = loss_fn_vgg.forward(pred, depth).mean() + l1_loss(pred, depth) + l2_loss(pred, depth)


if __name__ == '__main__':
    experiment_name = 'RGB+DP_upp_r50'
    rgb_input = True
    assert os.path.exists(f"./checkpoints/{experiment_name}"), "Experiment name does not exist"

    model = getModel(rgb_input)
    test(experiment_name, model, rgb_input)