import os, shutil, cv2
import numpy as np
import torch
from torchvision import transforms
# from unet import UNet,MC_LURE, CustomLoss4
from datasets import custom_test_dataset
import config as cfg
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from deepinv.models import DnCNN,DRUNet,SCUNet,GSDRUNet
from loss import MC_LURE
from model import UNet
import pytorch_msssim
np.random.seed(0)
res_dir = cfg.res_dir

if os.path.exists(res_dir):
    shutil.rmtree(res_dir)

if not os.path.exists(res_dir):
    os.mkdir(res_dir)
    
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

model = UNet(n_classes = 1, depth = cfg.depth, padding = True).to(device) # try decreasing the depth value if there is a memory error
# model=GSDRUNet(alpha=1.0, in_channels=1, out_channels=1, nb=2, nc=[64, 128, 256, 512], act_mode='E', pretrained=None, train=True, device='cuda:1')
# model=DRUNet(in_channels=1, out_channels=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', pretrained=None, train=False, device='cpu')
ckpt_path = os.path.join(cfg.models_dir, cfg.ckpt)
ckpt = torch.load(ckpt_path)
print(f'\nckpt loaded: {ckpt_path}')
model_state_dict = ckpt['model_state_dict']
model.load_state_dict(model_state_dict)
model.to(device)

def get_img_strip(tensr):
    bs, _ , h, w = tensr.shape
    tensr2np = (tensr.cpu().numpy().clip(0,1)*255).astype(np.uint8)    
    return tensr2np.squeeze(0).squeeze(0)

def denoise(noisy_imgs, out):
    out = get_img_strip(out)
    denoised = np.concatenate((noisy_imgs, out), axis = 0)
    return denoised 

def mona_post_processing(out1,org,noisy):
    A = torch.max(out1).item()
    B = torch.min(out1).item()
    out1 = out1.squeeze().squeeze().cpu()
    out1 = out1.numpy()
    out1 = (out1 - B) / (A-B) 
    out1 = out1 *255.0
    mse = np.mean((out1 - org) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    mse1=np.mean((noisy - org) ** 2)
    psnr1 = 20 * np.log10(255.0 / np.sqrt(mse1))
    ssim1 = ssim(org,noisy,data_range=255.0)
    ssim2 = ssim(org,out1,data_range=255.0)
    # print('MSE between noisy image and Original Image is :',mse1)
    print('PSNR between noisy image and Original Image is :',psnr1)
    print('SSIM between noisy image and Original Image is :',ssim1)
    # print('MSE between denoised image and Original Image is :',mse)
    print('PSNR between denoised image and Original Image is :',psnr)
    print('SSIM between denoised image and Original Image is :',ssim2)
    return psnr1,psnr,ssim1,ssim2,out1



print('\nDenoising noisy images...')
cnt = 0 
psnr_list_input = []
psnr_list_output = []
ssim_list_input = []
ssim_list_output = []

PSNR_Sum_Input = 0 
PSNR_Sum_Input1 = 0 
SSIM_Sum_Input = 0 
PSNR_Sum_Output = 0 
PSNR_Sum_Output1 = 0 
SSIM_Sum_Output = 0 
SSIM_Sum_Output1 = 0 
SSIM_Sum_Input1 = 0 
loss_fn1 = nn.MSELoss()
for i in os.listdir('../data/afhq/val/imgs'):
    img=cv2.imread(f"../data/afhq/val/imgs/{i}")
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # noisy_img=cv2.resize(img_gray,(512,512))
    cv2.imwrite(f"{cfg.res_dir}/act_{i}.jpg",img_gray)
    log_normal_noise = np.random.lognormal(mean=cfg.mu,sigma=cfg.sigma, size=img_gray.shape)
    noisy_img= np.copy(img_gray)
    noisy_img = noisy_img.astype(np.float32)
    noisy_img *= log_normal_noise
    k = 255.0 / np.max(noisy_img)
    noisy_img *= k 
    cv2.imwrite(f"{cfg.res_dir}/noisy_{i}.jpg",noisy_img)
    print(cfg.mu+k)
    y=noisy_img.copy()
    noisy_img=noisy_img.astype(np.uint8)
    noisy_img = transform(noisy_img)
    img_gray1 = transform(img_gray)
    noisy_img=torch.unsqueeze(noisy_img, dim=0)


    img_gray1=torch.unsqueeze(img_gray1, dim=0)

    with torch.no_grad():                                                                    
        noisy_img = noisy_img.to(device)
        img_gray1=img_gray1.to(device)
        out = model(noisy_img)
        mse2 = loss_fn1(out,img_gray1)
        psnr21 = 20 * np.log10(1.0 / np.sqrt(mse2.item()))
        psnr_list_output.append(psnr21)
        mse1 = loss_fn1(noisy_img,img_gray1)
        psnr11 = 20 * np.log10(1.0 / np.sqrt(mse1.item()))
        psnr_list_input.append(psnr11)
        ssimlib=pytorch_msssim.ssim(img_gray1,out, data_range=1.0)
        ssim_list_output.append(ssimlib.item())
        ssimin=pytorch_msssim.ssim(img_gray1,noisy_img, data_range=1.0)
        ssim_list_input.append(ssimin.item())
        psnr1,psnr2,ssim1,ssim2,denoised = mona_post_processing(out,img_gray,y)
        denoised2=get_img_strip(out)
        PSNR_Sum_Input += psnr11
        SSIM_Sum_Input += ssim1
        PSNR_Sum_Output += psnr21
        SSIM_Sum_Output += ssim2
        PSNR_Sum_Input1 += psnr1
        PSNR_Sum_Output1 += psnr2
        SSIM_Sum_Input1 += ssimin
        SSIM_Sum_Output1 += ssimlib
        cnt+=1 

        cv2.imwrite(f"./{cfg.res_dir}/Output{i}", denoised)
print('\n\nresults saved in \'{}\' directory'.format(res_dir))
print('Average PSNR of noisy images is :',PSNR_Sum_Input/cnt)
print('Average SSIM of noisy images is :',SSIM_Sum_Input1/cnt)
print('Average PSNR of denoised images is :',PSNR_Sum_Output/cnt)
print('Average SSIM of denoised images is :',SSIM_Sum_Output1/cnt)
print('Average PSNR of noisy images after post-processing :',PSNR_Sum_Input1/cnt)
print('Average SSIM of noisy images after post-processing :',SSIM_Sum_Input/cnt)
print('Average PSNR of denoised images after post-processing :',PSNR_Sum_Output1/cnt)
print('Average SSIM of denoised images after post-processing :',SSIM_Sum_Output/cnt)



print('\nFin.')


