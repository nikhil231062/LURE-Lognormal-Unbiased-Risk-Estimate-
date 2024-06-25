DATA:
    AFHQ: Download AFHQ dataset and place the train data in data/afhq/train/imgs folder. combine images in all the classes into imgs folder. Similarly place test data in data/afhq/val/imgs.
    Noisy Data: Contains noisy images corresponding to sigma of 0.2. Keep your noisy image in this folder inorder to estimate noise.

NOISE ESTIMATION : 
    Run noise_est.py files after replacing the path of the required noisy image. This uses median based approach for estimating the sigma of gaussian noise. Hence we take log of the image containing lognormal noise to convert noise to gaussian domain.

MODEL :
     This Folder contains weights of a UNet Model with (in_channels=out_channels=1) trained with Monte Carlo LURE loss function . The noise level (sigma) was set to 0.5 . This weights can be loaded in test.py to obtain denoising results for a dataset .

MAIN : 
    This Folder contains the important files for LURE Implementation which are 
     1) train.py : This file contains the main training loop with MC LURE as loss function , loss function can be changed to MSE for training with oracle MSE for comparasion purposes . The file load the data using datasets.py .
     2) datasets.py : This file has definition of class "DAE Dataset" which is used in train.py as well test.py . This dataset loads the clean image and then multiply the lognormal noise with sigma taken from config.py .
     3) config.py : This file contains the values of different hyperparameters which are used in other files . This contains learning rate , epochs , sigma , data directory , result directory etc.
     4) test.py : Trained model can be evaluated with SSIM and PSNR values using this file . Make sure to change the directory in the code .
     5) loss.py : This file contains the implementation of Monte Carlo LURE and Mean Square Error loss function . Both of the functions take pytorch tensors as input and return the pytorch tensor (scalar) as output . You cam just take this file and use with your custom code as well .

NOTE :- There are no implementations of model as model architecture have been taken directly from deepinverse library . 


requirements.txt : contain all the required libraries with there required versions . 