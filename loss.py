#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothLoss(nn.Module):
    def __init__(self, device):
        super(SmoothLoss, self).__init__()
       
        gradx = torch.FloatTensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]).to(device)
        grady = torch.FloatTensor([[-1, -2, -1],
                                   [0,   0,  2],
                                   [1,   0,  1]]).to(device)
        
        self.disp_gradx_ry = gradx.unsqueeze(0).unsqueeze(0)
        self.disp_grady_ry = grady.unsqueeze(0).unsqueeze(0)
        self.disp_gradx = self.disp_gradx_ry.repeat(1, 128, 1, 1) # NOTE: batch_size is hard coded
        self.disp_grady = self.disp_grady_ry.repeat(1, 128, 1, 1) # batch_size is hard coded
        # print(self.disp_gradx.shape, self.disp_grady.shape)
        self.img_gradx = self.disp_gradx_ry.repeat(128, 1, 1, 1) # NOTE: batch_size is hard coded
        self.img_grady = self.disp_grady_ry.repeat(128, 1, 1, 1) # batch_size is hard coded
        # print(self.img_gradx.shape, self.img_grady.shape)
        # self.min_depth = 15.
        # self.max_depth = 75.


    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        grad_disp_x = torch.abs(F.conv2d(disp, self.disp_gradx, padding=1, stride=1))
        grad_disp_y = torch.abs(F.conv2d(disp, self.disp_grady, padding=1, stride=1))

        grad_img_x = torch.abs(torch.mean(F.conv2d(img, self.img_gradx, padding=1, stride=1), dim=1, keepdim=True))
        grad_img_y = torch.abs(torch.mean(F.conv2d(img, self.img_grady, padding=1, stride=1), dim=1, keepdim=True))

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        loss_x = 10 * (torch.sqrt(torch.var(grad_disp_x) + 0.15 * torch.pow(torch.mean(grad_disp_x), 2)))
        loss_y = 10 * (torch.sqrt(torch.var(grad_disp_y) + 0.15 * torch.pow(torch.mean(grad_disp_y), 2)))

        return loss_x + loss_y

    

    def forward(self, rgb, disp):
        N, C, H, W = rgb.shape
        # disp = self.compute_disp(depth)
        loss = self.get_smooth_loss(disp, rgb)
        return loss

class pyramid_SIDL(nn.Module):
    def __init__(self):
        super(pyramid_SIDL, self).__init__()
        self.n = 4 # number of pyramid levels

    def build_pyramid(self, img, n):
        if len(img.shape) == 3:
            img = img.unsqueeze(1)
        pyramid = [img]
        h = img.shape[2]
        w = img.shape[3]
        for i in range(n - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            pyramid.append(F.interpolate(pyramid[i], (nh, nw), mode='bilinear', align_corners=True))
            # print(pyramid[i].shape)
        return pyramid

    def x_grad(self, img):
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        img = F.pad(img, (1, 0, 0, 0), mode='replicate')
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        return grad_x

    def y_grad(self, img):
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        img = F.pad(img, (0, 0, 0, 1), mode='replicate')
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return grad_y

    def disp_smoothness(self, disp, pyramid):

        disp_x_grad = [self.x_grad(i) for i in disp]
        disp_y_grad = [self.y_grad(j) for j in disp]

        image_x_grad = [self.x_grad(i) for i in pyramid]
        image_y_grad = [self.y_grad(j) for j in pyramid]

        #e^(-|x|) weights, gradient negatively exponential to weights
        #average over all pixels in C dimension
        weights_x = [torch.exp(-torch.mean(torch.abs(i), 1, keepdim=True)) for i in image_x_grad]
        weights_y = [torch.exp(-torch.mean(torch.abs(j), 1, keepdim=True)) for j in image_y_grad]

        smoothness_x = []
        for i in range(self.n):
            # print(disp_x_grad[i].shape, weights_x[i].shape)
            element_wise = disp_x_grad[i] * weights_x[i]
            smoothness_x.append(element_wise)

        smoothness_y = [disp_y_grad[j] * weights_y[j] for j in range(self.n)]

        smoothness = [torch.mean(torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])) / 2 ** i for i in range(self.n)]

        return smoothness
    
    def forward(self, rgb, disp):
        N, C, H, W = rgb.shape
        pyramid = self.build_pyramid(rgb, self.n)
        disp_pyramid = self.build_pyramid(disp, self.n)
        loss = self.disp_smoothness(disp_pyramid, pyramid)
        return sum(loss)

# sidl = pyramid_SIDL().to('cuda')
# sidl_loss = sidl(torch.rand(64, 3, 224, 224), torch.rand(64, 224, 224))
# print(sidl_loss)
# class Hessian2DNorm():
#     def __init__(self):
#         pass
#     def __call__(self, img):
#         # Compute Individual derivatives
#         fxx = img[..., 1:-1, :-2] + img[..., 1:-1, 2:] - 2*img[..., 1:-1, 1:-1]
#         fyy = img[..., :-2, 1:-1] + img[..., 2:, 1:-1] - 2*img[..., 1:-1, 1:-1]
#         fxy = img[..., :-1, :-1] + img[..., 1:, 1:] - \
#               img[..., 1:, :-1] - img[..., :-1, 1:]
          
#         return torch.sqrt(fxx.abs().pow(2) +\
#                           2*fxy[..., :-1, :-1].abs().pow(2) +\
#                           fyy.abs().pow(2)).sum()


# class Hessian3DNorm():
#     def __init__(self):
#         pass
#     def __call__(self, img):
#         # Compute Individual derivatives
#         fxx = img[...,1:-1, 1:-1, :-2] + img[...,1:-1, 1:-1, 2:] - 2*img[...,1:-1, 1:-1, 1:-1]
#         fyy = img[...,1:-1, :-2, 1:-1] + img[...,1:-1, 2:, 1:-1] - 2*img[...,1:-1, 1:-1, 1:-1]
#         fxy = img[...,1:-1, :-1, :-1] + img[...,1:-1, 1:, 1:] - \
#                 img[...,1:-1, 1:, :-1] - img[...,1:-1, :-1, 1:]
#         fzz = img[...,:-2, 1:-1, 1:-1] + img[...,2:, 1:-1, 1:-1] - 2*img[...,1:-1, 1:-1, 1:-1]
#         fxz = img[...,:-1, 1:-1, :-1] + img[...,1:, 1:-1, 1:] - \
#                 img[...,1:, 1:-1, :-1] - img[...,:-1, 1:-1, 1:]
#         fyz = img[...,:-1, :-1, 1:-1] + img[...,1:, 1:, 1:-1] - \
#                 img[...,1:, :-1, 1:-1] - img[...,:-1, 1:, 1:-1]
          
#         return torch.sqrt(fxx.abs().pow(2) +\
#                           2*fxy[..., :-1, :-1].abs().pow(2) +\
#                           fyy.abs().pow(2) + fzz.abs().pow(2) +\
#                           2*fxz[...,:-1, :, :-1].abs().pow(2) + 2*fyz[...,:-1,:-1,:].abs().pow(2) ).sum()


# class TV2DNorm():
#     def __init__(self, mode='l1'):
#         self.mode = mode
#     def __call__(self, img):
#         grad_x = img[..., 1:, 1:] - img[..., 1:, :-1]
#         grad_y = img[..., 1:, 1:] - img[..., :-1, 1:]
        
#         if self.mode == 'isotropic':
#             #return torch.sqrt(grad_x.abs().pow(2) + grad_y.abs().pow(2)).mean()
#             return torch.sqrt(grad_x**2 + grad_y**2).sum()
#         elif self.mode == 'l1':
#             return abs(grad_x).sum() + abs(grad_y).sum()
#         elif self.mode == 'hessian':
#             return Hessian2DNorm()(img)
#         else:
#             return (grad_x.pow(2) + grad_y.pow(2)).sum()     


# class TV3DNorm():
#     def __init__(self, mode='l1'):
#         self.mode = mode
#     def __call__(self, img):
#         grad_x = img[...,1:, 1:, 1:] - img[...,1:, 1:, :-1]
#         grad_y = img[...,1:, 1:, 1:] - img[...,1:, :-1, 1:]
#         grad_z = img[...,1:, 1:, 1:] - img[...,:-1, 1:, 1:]
        
#         if self.mode == 'isotropic':
#             #return torch.sqrt(grad_x.abs().pow(2) + grad_y.abs().pow(2)).mean()
#             return torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2).sum()
#         elif self.mode == 'l1':
#             return abs(grad_x).sum() + abs(grad_y).sum() + abs(grad_z).sum() 
#         elif self.mode == 'hessian':
#             return Hessian3DNorm()(img)
#         else:
#             return (grad_x.pow(2) + grad_y.pow(2) + grad_z.pow(2)).sum()     

