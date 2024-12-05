import os

# path to saving models
models_dir = '/home/gayathri/LURE-Lognormal-Unbiased-Risk-Estimate-/model/'
# path to saving loss plots
losses_dir = 'losses'

# path to the data directories
data_dir = '/home/gayathri/LURE-Lognormal-Unbiased-Risk-Estimate-/data/afhq'
train_dir = 'train/'
val_dir = 'val/'
imgs_dir = 'cat'
noisy_dir = 'noisy'
debug_dir = 'debug'

# depth of UNet 
depth = 4 # try decreasing the depth value if there is a memory error

# text file to get text from
txt_file_dir = 'shitty_text.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 18000
train_percentage = 0.8

# epsilon = 0.5
resume = False  # False for trainig from scratch, True for loading a previouslys saved weight
ckpt='best_mure_model.pth' # model file path to load the weights from, only useful when resume is True

lr = 1e-6      # learning rate
epochs = 10    # epochs to train for 

sigma=0.3
mu=0.5

# batch size for train and val loaders
batch_size = 1 # try decreasing the batch_size if there is a memory error

# log interval for training and validation
log_interval = 25

test_dir = os.path.join(data_dir, val_dir, noisy_dir)
res_dir = '/home/gayathri/LURE-Lognormal-Unbiased-Risk-Estimate-/outputs/'
test_bs = 64
