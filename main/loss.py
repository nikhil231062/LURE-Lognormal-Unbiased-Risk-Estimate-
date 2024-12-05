import torch
from torch import nn
import torch.nn.functional as F
import config as cfg
import numpy as np

    
class MC_LURE(nn.Module):
    def __init__(self):
        super(MC_LURE, self).__init__()
    
    def forward(self, denoised_img, noisy_img, sigma, mu,model,device,k):    
        epsilon_list=np.linspace(0.05,0.95,12)
        # pertua=torch.zeros((noisy_img.shape[2],noisy_img.shape[3])).to(device)
        # pertub=torch.zeros((noisy_img.shape[2],noisy_img.shape[3])).to(device)
        loss=0
        for i in range(noisy_img.shape[0]):
            mu1 = mu + torch.log(k[i])
            exp_term1 = torch.exp(-2 * (sigma**2 + mu1))
            exp_term2 = torch.exp(-0.5 * sigma**2 - mu1)
            norm_noisy_img = torch.norm(noisy_img[i][0], p="fro")
            norm_denoised_img= torch.norm(denoised_img[i][0], p="fro")
            y = noisy_img[i][0].clone().requires_grad_()
            b = torch.randn(noisy_img[i][0].shape).to(device)
            ydotb = y*b
            dot_product2 = 0 
            for epsilon in epsilon_list:
                pertua=(noisy_img[i][0]+epsilon*ydotb)
                pertub=(noisy_img[i][0]-epsilon*ydotb)
                pertubate1=model(pertua.unsqueeze(0).unsqueeze(0))
                pertubate2=model(pertub.unsqueeze(0).unsqueeze(0))
                dot_product2_inter = torch.dot(torch.flatten(ydotb),torch.flatten(pertubate1.squeeze(0).squeeze(0)-pertubate2.squeeze(0).squeeze(0)))
                dot_product2_inter = dot_product2_inter/(2*epsilon*noisy_img[i][0].shape[0]*noisy_img[i][0].shape[1])
                dot_product2 = dot_product2 + dot_product2_inter
            dot_product2 = dot_product2 / 12.0 

            dot_product1 = torch.dot(torch.flatten(noisy_img[i][0]),torch.flatten(denoised_img[i][0]))
            loss += (exp_term1 * (norm_noisy_img**2) + (norm_denoised_img**2) - 2 * exp_term2 * (dot_product1- (sigma**2)*dot_product2))/(noisy_img[i][0].shape[0]*noisy_img[i][0].shape[1])
        return (loss / (noisy_img.shape[0] * noisy_img.shape[1]))**2
    
class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()

    def forward(self, denoised_img, img):
        loss=0
        for i in range(img.shape[0]):
            norm_img = torch.norm(img[i][0], 'fro')
            norm_denoised_img = torch.norm(denoised_img[i][0], 'fro')

            dot_product = torch.dot(torch.flatten(img[i][0]), torch.flatten(denoised_img[i][0]))

            loss += ((norm_img**2) + (norm_denoised_img**2) - 2 * dot_product)/(img[i][0].shape[0]**2)
        return (loss/(img.shape[0] * img.shape[1])) **2
