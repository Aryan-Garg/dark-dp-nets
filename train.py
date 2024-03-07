#!/usr/bin/env python3

import os
import numpy as np
import cv2
import argparse
import random
import time

import torch
import lpips
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 

from torch.utils.data import Dataset, DataLoader

from dataset import BroDataset, depthEstimationDataset
from loss import pyramid_SIDL
from metrics import compute_report_errors
from torchsummary import summary
import wandb


def getModel(args):
    # NOTE - To Try: 
    # 1. resnet50 (inspired from DeepLens)

    if args.input_type == 'rgb_dp':
        in_chans = 5
    elif args.input_type == 'rgb':
        in_chans = 3
    elif args.input_type == 'dp':
        in_chans = 2

    model = smp.Unet(
        encoder_name=args.model, # choose deployable models 
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        decoder_use_batchnorm= False if "resnet" in args.model else True,    # don't use batchnorm for decoder (results in droplet artifacts for ResNet50 backbone)
        decoder_attention_type=None,    # don't use attention layers in decoder. Options: [None, scse] # NOTE: Experiment?
        in_channels=in_chans,           # model input channels (2 for dp channels + 3 for RGB)
        classes=1,                      # model output channels (disp is 1 channel)
        activation=None,                # regress the disparity directly; no last layer activation
    )

    # summary(model.to(device), (in_chans, 224, 224))
    
    return model


# TODO: experiments 2.X: train with AdamW and no BN in decoder
# TODO: experiments 3.X: add smoothness loss on prediction
def getOptimizer(args, model):
    if "resnet" in args.model:
        # standard out of the box
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    elif "mobile" in args.model:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum, eps=args.eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=5,
                                                           factor=0.1, 
                                                           mode='min')
    
    return optimizer, scheduler


def train_val_test(args, experiment_name, model, opt, sch, inp_type, device):
    epochs = args.epochs
    val_every_epoch = args.val_every_epoch

    # train_loader = DataLoader(BroDataset(txt_files, mode='train', rgb_input=rgb_input), batch_size=6, shuffle=True)
    # val_loader = DataLoader(BroDataset(txt_files, mode='val', rgb_input=rgb_input), batch_size=6, shuffle=True)
    train_loader = DataLoader(depthEstimationDataset(args.train_files, mode = "train", input_type=inp_type), 
                              batch_size=args.batch_size, 
                              shuffle=True)
    val_loader = DataLoader(depthEstimationDataset(args.val_files, mode = "val", input_type=inp_type), 
                            batch_size=args.batch_size, 
                            shuffle=False)

    model.to(device)

    best_val_loss = np.inf
    # lpips_lambda = 0.3
    smooth_lambda = 1.

    l1_loss = torch.nn.L1Loss().to(device)

    useL2 = False
    if useL2:
        l2_loss = torch.nn.MSELoss().to(device)

    useSmooth = False
    if useSmooth:
        pyramid_sidl = pyramid_SIDL().to(device)

    for epoch in range(epochs):

        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, sample in pbar:
            opt.zero_grad()

            # progressive_lambda = 1. * i / len(train_loader) # first l2 then l1
            progressive_lambda = 1. # l2 = 0.2 l1 = 0.8

            if input_type == 'rgb_dp':
                dp_input, disparity = sample[:, :5, :, :], sample[:, 5, :, :]
            elif input_type == 'rgb':
                dp_input, disparity = sample[:, :3, :, :], sample[:, 3, :, :]
            elif input_type == 'dp':
                dp_input, disparity = sample[:, :2, :, :], sample[:, 2, :, :]
            
            # print(dp_input.shape, disparity.shape)
            dp_input, disparity = dp_input.to(device), disparity.squeeze().to(device)
            
            pred = model(dp_input).squeeze()
            # print(pred.shape, disparity.shape)
            # print(torch.max(pred), torch.min(pred))
            
            if useSmooth:
                loss_smooth = smooth_lambda * pyramid_sidl(dp_input[:,:3, :, :], pred) 
            else:
                loss_smooth = torch.Tensor([0.]).to(device)
            # loss_lpips = lpips_lambda * loss_fn_vgg.forward(torch.stack((pred, pred, pred), dim=1), 
            #                                                 torch.stack((disparity, disparity, disparity), dim = 1)).sum() 
            loss_l1 = progressive_lambda * l1_loss(pred, disparity)  
            if useL2:
                loss_l2 = 0.1 * (1. - progressive_lambda) * l2_loss(pred, disparity) 
            else:
                loss_l2 = torch.Tensor([0.]).to(device)

            loss = 1. * (loss_l1 + loss_l2 + loss_smooth) # + loss_lpips
            
            loss.backward()
            
            opt.step()
            sch.step(loss.item())

            if i % 50 == 0:
                wandb.log({"epoch": epoch+1,
                            "l1_loss": loss_l1.item(),
                            "l2_loss": loss_l2.item(),
                            "smooth_loss": loss_smooth.item(),
                            # "lpips_loss": loss_lpips.item(),
                            "total_loss": loss.item()})

            pbar.set_description(f"Ep: {epoch+1} | pl: {progressive_lambda:.3f} | Loss: {loss.item():.3f}")
            
        
        if (epoch+1) % val_every_epoch == 0:
            this_epoch_val_loss = 0
            pbar2 = tqdm(enumerate(val_loader), total=len(val_loader))
            for i, sample in pbar2:
                if input_type == 'rgb_dp':
                    dp_input, disparity = sample[:, :5, :, :], sample[:, 5, :, :]
                elif input_type == 'rgb':
                    dp_input, disparity = sample[:, :3, :, :], sample[:, 3, :, :]
                elif input_type == 'dp':
                    dp_input, disparity = sample[:, :2, :, :], sample[:, 2, :, :]
            
                dp_input, disparity = dp_input.to(device), disparity.squeeze().to(device)
                pred = model(dp_input).squeeze()

                if useSmooth:
                    loss_smooth = smooth_lambda * pyramid_sidl(dp_input[:,:3,:,:], pred)
                else:
                    loss_smooth = torch.Tensor([0.]).to(device)
                # loss_lpips = lpips_lambda * loss_fn_vgg.forward(torch.stack((pred, pred, pred), dim=1), 
                #                                             torch.stack((disparity, disparity, disparity), dim = 1)).sum()
                loss_l1 = progressive_lambda * l1_loss(pred, disparity)   
                if useL2:
                    loss_l2 = 0.1 * (1. - progressive_lambda) * l2_loss(pred, disparity) 
                else:
                    loss_l2 = torch.Tensor([0.]).to(device)

                loss = 1. * (loss_l1 + loss_l2 + loss_smooth) # + loss_lpips

                if i % 25 == 0:
                    wandb.log({"val/l1_loss": loss_l1.item(),
                        "val/l2_loss": loss_l2.item(),
                        "val/smooth_loss": loss_smooth.item(),
                        # "val/lpips_loss": loss_lpips.item(),
                        "val/total_loss": loss.item()})
                    wandb.log(compute_report_errors(disparity.cpu().numpy(), pred.cpu().detach().numpy())) # inst

                this_epoch_val_loss += loss.item()

            this_epoch_val_loss /= len(val_loader)
            
            if best_val_loss > this_epoch_val_loss: # NOTE: save all models
                best_val_loss = min(best_val_loss, this_epoch_val_loss)
                torch.save(model.state_dict(), f'./checkpoints/{experiment_name}/{epoch+1}.pth')
                print("Best val loss:", best_val_loss)
                # Save last best pred-gt pair
                for i in range(pred.shape[0]):
                    image = wandb.Image(10*pred[i,:,:].detach().cpu().numpy(), caption=f"predicted_disp_{i}")
                    gt = wandb.Image(10*disparity[i,:,:].cpu().numpy(), caption=f"gt_disp_{i}")
                    # log side by side
                    wandb.log({"best_val/predicted_disp": image, "best_val/gt_disp": gt})


        elif epoch + 1 == epochs:
            ckpts = os.listdir(f'./checkpoints/{experiment_name}')
            ckpt = torch.load(f'./checkpoints/{experiment_name}/{ckpts[-1]}', map_location='cpu')
            model.load_state_dict(ckpt)
            model.to(device)
            
            test_loader = val_loader
            final_metrics = {"abs_rel": 0, "sq_rel": 0, "rmse": 0,
                            "log_rmse": 0, "del_1": 0, "del_2": 0, "del_3": 0}
            for epoch in range(1):
                pbar = tqdm(enumerate(test_loader), total=len(test_loader))
                for i, sample in pbar:
                    if input_type == 'rgb_dp':
                        dp_input, disparity = sample[:, :5, :, :], sample[:, 5, :, :]
                    elif input_type == 'rgb':
                        dp_input, disparity = sample[:, :3, :, :], sample[:, 3, :, :]
                    elif input_type == 'dp':
                        dp_input, disparity = sample[:, :2, :, :], sample[:, 2, :, :]
            

                    dp_input, disparity = dp_input.to(device), disparity.squeeze().to(device)
                    with torch.no_grad():
                        pred = model(dp_input).squeeze()
                        if useSmooth:
                            loss_smooth = smooth_lambda * pyramid_sidl(dp_input[:,:3,:,:], pred)
                        else:
                            loss_smooth = torch.Tensor([0.]).to(device)
                        # loss_lpips = lpips_lambda * loss_fn_vgg.forward(torch.stack((pred, pred, pred), dim=1), 
                        #                                         torch.stack((disparity, disparity, disparity), dim = 1)).sum()
                        loss_l1 = progressive_lambda * l1_loss(pred, disparity) 
                        if useL2:
                            loss_l2 = 0.1 * (1. - progressive_lambda) * l2_loss(pred, disparity) 
                        else:
                            loss_l2 = torch.Tensor([0.]).to(device)

                        loss = 1. * (loss_l1 + loss_l2 + loss_smooth)  # + loss_lpips

                        wandb.log({"test/l1_loss": loss_l1.item()})
                        if useL2:
                            wandb.log({"test/l2_loss": loss_l2.item()})
                        if useSmooth:
                            wandb.log({"test/smooth_loss": loss_smooth.item()})
                        # wandb.log({"test/lpips_loss": loss_lpips.item()})
                        wandb.log({"test/total_loss": loss.item()})

                        pred = pred.detach().cpu()
                        for i in range(pred.shape[0]):
                            image = wandb.Image(pred[i,0,:,:].numpy(), caption=f"predicted_disp_{i}")
                            gt = wandb.Image(disparity[i,0,:,:].numpy(), caption=f"gt_disp_{i}")
                            # log side by side
                            wandb.log({"test/predicted_disp": image, "test/gt_disp": gt})
                            # plt.imsave(f'./save_{experiment_name}/{i+1}_pred.png', *255)
                            # plt.imsave(f'./save_{experiment_name}/{i+1}_dp_left.png', dp_input[i,0,:,:].cpu().numpy()*255)
                            # plt.imsave(f"./save_{experiment_name}/{i+1}_gt.png", disp[i,0,:,:].cpu().numpy()*255)

                        metrics = compute_report_errors(disparity.cpu().numpy(), pred.cpu().detach().numpy())
                        for k in final_metrics.keys():
                            final_metrics[k] += metrics[k]
                        
                print(final_metrics / len(test_loader))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--experiment_name", "-exp", type=str, default="unicornX")

    args.add_argument("--train_files", "-tf", type=str, default="./files/skip10_train_files.txt")
    args.add_argument("--val_files", "-vf", type=str, default="./files/skip10_val_files.txt")
        
    args.add_argument("--input_type", type=str, default="rgb_dp")
    args.add_argument("--batch_size", "-bs", type=int, default=32) # never keep bs = 1

    args.add_argument("--epochs", "-e", type=int, default=20)
    args.add_argument("--val_every_epoch", type=int, default=1)

    args.add_argument("--model", type=str, default='mobilenet_v2')

    args.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    args.add_argument("--momentum", type=float, default=0.9)
    args.add_argument("--weight_decay", type=float, default=0.01)
    args.add_argument("--eps", type=float, default=1e-7)
    args.add_argument("--device_num", "-dn", type=int, default=0)

    args = args.parse_args()

    device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else 'cpu')

    # loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)


    experiment_name = args.experiment_name
    wandb.init(project='distill_2_mobile_dp', 
            config={"learning_rate": args.learning_rate, 
                    "input_type": args.input_type,
                    "momentum": args.momentum, 
                    "weight_decay": args.weight_decay, 
                    "eps": args.eps,
                    "batch_size": args.batch_size, 
                    "epochs": args.epochs, 
                    "val_every_epoch": args.val_every_epoch, 
                    "model": args.model})
    
    input_type = args.input_type
    wandb.run.name = f"{experiment_name}"

    if not os.path.exists(f"./checkpoints/{experiment_name}"):
        os.makedirs(f"./checkpoints/{experiment_name}")

    model = getModel(args)

    opt, sch = getOptimizer(args, model)
    train_val_test(args, experiment_name, model, opt, sch, input_type, device)