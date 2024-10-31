import sys, os, time, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend('agg') # for servers not supporting display
from skimage.metrics import structural_similarity as ssim
# import neccesary libraries for defining the optimizers
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
# from loss_func import MC_LURE,MSE_Loss
from loss import MC_LURE,MSE_Loss
from model import UNet
from datasets import DAE_dataset
import config as cfg
from pytorch_msssim import ssim
import csv 
from deepinv.models import DnCNN,DRUNet,SwinIR,SCUNet,GSDRUNet
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)

script_time = time.time()

def q(text = ''):
    print('> {}'.format(text))
    sys.exit()

data_dir = cfg.data_dir
train_dir = cfg.train_dir
val_dir = cfg.val_dir
    
models_dir = cfg.models_dir
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

losses_dir = cfg.losses_dir
if not os.path.exists(losses_dir):
    os.mkdir(losses_dir)

def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

def plot_losses(running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss, epoch):
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('epoch train loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('epoch train loss')
    ax1.plot(train_epoch_loss)
    
    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(val_epoch_loss)
 
    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(running_train_loss)

    ax4.title.set_text('batch val loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch val loss')
    ax4.plot(running_val_loss)
    
    plt.savefig(os.path.join(losses_dir,'losses_{}.png'.format(str(epoch + 1).zfill(2))))

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

# print(os.path.join(data_dir, train_dir))
train_dataset       = DAE_dataset(os.path.join(data_dir, train_dir), transform = transform)
val_dataset         = DAE_dataset(os.path.join(data_dir, val_dir), transform = transform)

print('\nlen(train_dataset) : ', len(train_dataset))
print('len(val_dataset)   : ', len(val_dataset))

batch_size = cfg.batch_size

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)

print('\nlen(train_loader): {}  @bs={}'.format(len(train_loader), batch_size))
print('len(val_loader)  : {}  @bs={}'.format(len(val_loader), batch_size))

# defining the model
model = UNet(n_classes = 1, depth = cfg.depth, padding = True).to(device) # try decreasing the depth value if there is a memory error
# model=DRUNet(in_channels=1, out_channels=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', pretrained=None, train=True, device='cuda:1')
# model=GSDRUNet(alpha=1.0, in_channels=1, out_channels=1, nb=2, nc=[64, 128, 256, 512], act_mode='E', pretrained=None, train=True, device='cuda:1')

resume = cfg.resume

if not resume:
    print('\nfrom scratch')
    train_epoch_loss = []
    val_epoch_loss = []
    running_train_loss = []
    running_val_loss = []
    epochs_till_now = 0
else:
    ckpt_path = os.path.join(models_dir, cfg.ckpt)
    ckpt = torch.load(ckpt_path)
    print(f'\nckpt loaded: {ckpt_path}')
    model_state_dict = ckpt['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.to(device)
    losses = ckpt['losses']
    running_train_loss = losses['running_train_loss']
    running_val_loss = losses['running_val_loss']
    train_epoch_loss = losses['train_epoch_loss']
    val_epoch_loss = losses['val_epoch_loss']
    epochs_till_now = ckpt['epochs_till_now']

lr = cfg.lr
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
loss_fn = MC_LURE()
loss_fn1 = MSE_Loss()
# loss_fn2=nn.MSELoss()
log_interval = cfg.log_interval
epochs = cfg.epochs

###
print('\nmodel has {} M parameters'.format(count_parameters(model)))
print(f'\nloss_fn        : {loss_fn}')
print(f'lr             : {lr}')
print(f'epochs_till_now: {epochs_till_now}')
print(f'epochs from now: {epochs}')
###

best_val_MSE_loss= 9999999
best_val_MURE_loss= 9999999
best_SSIM=-1
best_MSE_model=None
best_MURE_model=None
train_mse_loss =[]
train_mure_loss =[]
val_mure_loss =[]
val_mse_loss = []
val_ssim_list = []
# train_epoch_loss =[]


for epoch in range(epochs_till_now, epochs_till_now+epochs):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, epochs_till_now + epochs))    
    print('\nTRAINING...')
    epoch_train_start_time = time.time()
    running_mure_loss=[]
    running_mse_loss=[]
    running_val_mure_loss=[]
    running_val_mse_loss=[]
    running_ssim=[]

    for batch_idx, (imgs, noisy_imgs,k) in enumerate(tqdm(train_loader)):
        batch_start_time = time.time()
        imgs = imgs.to(device)
        noisy_imgs = noisy_imgs.to(device)

        optimizer.zero_grad()
        out = model(noisy_imgs)
        loss = loss_fn(out, noisy_imgs,sigma=torch.tensor(cfg.sigma),mu=torch.tensor(cfg.mu),model=model,device=device,k=k)
        mse_loss=loss_fn1(out,imgs)

        running_mse_loss.append(mse_loss.item())
        running_mure_loss.append(loss.item())

        loss.backward()
        # mse_loss.backward()
        optimizer.step()

        if (batch_idx + 1)%log_interval == 0:
            batch_time = time.time() - batch_start_time
            m,s = divmod(batch_time, 60)
            print('train loss @batch_idx {}/{}: {} in {} mins {} secs (per batch)'.format(str(batch_idx+1).zfill(len(str(len(train_loader)))), len(train_loader), loss.item(), int(m), round(s, 2)))
            print('NN_mse loss is :',mse_loss.item())
            print('MURE is :',loss.item())
        
    # train_epoch_loss.append(np.array(running_train_loss).mean())
    train_mse_loss.append(np.array(running_mse_loss).mean())
    train_mure_loss.append(np.array(running_mure_loss).mean())

    epoch_train_time = time.time() - epoch_train_start_time
    m,s = divmod(epoch_train_time, 60)
    h,m = divmod(m, 60)
    print('\nepoch train time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))

    print('\nVALIDATION...')
    epoch_val_start_time = time.time()
    # model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, noisy_imgs,k) in enumerate(val_loader):

            imgs = imgs.to(device)
            noisy_imgs = noisy_imgs.to(device)

            out = model(noisy_imgs)
            mse_loss = loss_fn1(out,imgs)
            print("The MSE is :" ,mse_loss )
            loss = loss_fn(out, noisy_imgs,torch.tensor(cfg.sigma),torch.tensor(cfg.mu),model,device,k)
            print("The MC LURE is :" ,loss )
            ssim1=ssim(imgs,out, data_range=1.0, size_average=True)
            running_val_mse_loss.append(mse_loss.item())
            running_val_mure_loss.append(loss.item())
            running_ssim.append(ssim1.item())

            if (batch_idx + 1)%log_interval == 0:
                print('val loss   @batch_idx {}/{}: {}'.format(str(batch_idx+1).zfill(len(str(len(val_loader)))), len(val_loader), loss.item()))
                print('mse loss: ',mse_loss.item())
    val_mse = np.array(running_val_mse_loss).mean()
    val_loss = np.array(running_val_mure_loss).mean()
    val_mse_loss.append(np.array(running_val_mse_loss).mean())
    val_mure_loss.append(np.array(running_val_mure_loss).mean())
    
    val_ssim = np.array(running_ssim).mean()
    val_ssim_list.append(np.array(running_ssim).mean())


    # Transpose the lists into rows
    rows = zip(train_mse_loss,train_mure_loss,val_mse_loss,val_mure_loss,val_ssim_list)

    # Specify the file name
    csv_file = "losses/losses_lure_sar.csv"

    # Write the rows to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Train_MSE", "Train_MURE", "Val_MSE", "Val_MURE","SSIM"])  # Header row
        writer.writerows(rows)

    if best_val_MURE_loss > val_loss:
        best_val_MURE_loss=val_loss
        best_MURE_model={'model_state_dict': model.state_dict(),
                'losses': {'mure_train_loss': train_mure_loss, 
                           'mure_val_loss': val_mure_loss, 
                           'mse_train_loss': train_mse_loss, 
                           'mse_val_loss': val_mse_loss,
                           'ssim':val_ssim_list},
                'epochs_till_now': epoch+1}
    if best_val_MSE_loss > val_mse:
        best_val_MSE_loss=val_mse
        best_MSE_model={'model_state_dict': model.state_dict(),
                'losses': {'mure_train_loss': train_mure_loss, 
                           'mure_val_loss': val_mure_loss, 
                           'mse_train_loss': train_mse_loss, 
                           'mse_val_loss': val_mse_loss,
                           'ssim':val_ssim_list},
                'epochs_till_now': epoch+1}
    if best_SSIM < val_ssim:
        best_SSIM=val_ssim
        best_SSIM_model={'model_state_dict': model.state_dict(), 
                'losses': {'mure_train_loss': train_mure_loss, 
                           'mure_val_loss': val_mure_loss, 
                           'mse_train_loss': train_mse_loss, 
                           'mse_val_loss': val_mse_loss,
                           'ssim':val_ssim_list},
                'epochs_till_now': epoch+1}
                
    
    epoch_val_time = time.time() - epoch_val_start_time
    m,s = divmod(epoch_val_time, 60)
    h,m = divmod(m, 60)
    print('\nepoch val   time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))

    # plot_losses(running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss,  epoch)   

    torch.save({'model_state_dict': model.state_dict(), 
                'epochs_till_now': epoch+1}, 
                os.path.join(models_dir, 'model{}.pth'.format(str(epoch + 1).zfill(2))))


torch.save(best_MSE_model, 
                os.path.join(models_dir, 'best_mse_model.pth'))
torch.save(best_MURE_model, 
                os.path.join(models_dir, 'best_mure_model.pth'))
torch.save(best_SSIM_model, 
                os.path.join(models_dir, 'best_ssim_model.pth'))

plt.figure()
# x = np.linspace(0,11,1)
plt.plot(train_mse_loss,'r',label='Train_MSE')
plt.plot(train_mure_loss,'b',label='Train_MURE') 
plt.plot(val_mse_loss,'g',label='Val_MSE')
plt.plot(val_mure_loss,'y',label='Val_MURE')
plt.legend()
plt.savefig(f'./plots/unet_lure_mar.png')



total_script_time = time.time() - script_time
m, s = divmod(total_script_time, 60)
h, m = divmod(m, 60)
print(f'\ntotal time taken for running this script: {int(h)} hrs {int(m)} mins {int(s)} secs')
  
print('\nFin.')
