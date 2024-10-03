DATA:
   AFHQ: Download AFHQ dataset and place the train data in data/afhq/train/imgs folder. combine images in all the classes into imgs folder. Similarly place test data in data/afhq/val/imgs.
   Noisy Data: Contains noisy images corresponding to a sigma of 0.2. Keep your noisy image in this folder in order to estimate noise.
   To download complete AFHQ dataset run this bash command : 
    `bash download_afhq.sh`


NOISE ESTIMATION :
   Run noise_est.py files after replacing the path of the required noisy image. This uses median based approach for estimating the sigma of Gaussian noise. Hence we take the log of the image containing lognormal noise to convert the noise to Gaussian domain.

   To run the code use this bash command by providing image path link . 
   For eg :  `python noise_est.py --img_path ../data/noisy_data/noisy_champs.png`


MODEL :
    This Folder contains weights of a UNet Model with (in_channels=out_channels=1) trained with the Monte Carlo LURE loss function. The noise level (sigma) was set to 0.5. These weights can be loaded in test.py to obtain denoising results for a dataset.


MAIN :
   This Folder contains the important files for LURE Implementation which are
    1) train.py: This file contains the main training loop with MC LURE as loss function, loss function can be changed to MSE for training with Oracle MSE for comparison purposes. The file loads the data using datasets.py.
    2) datasets.py: This file has the definition of the class "DAE Dataset" which is used in train.py as well as test.py. This dataset loads the clean image and then multiplies the lognormal noise with sigma taken from config.py.
    3) config.py: This file contains the values of different hyperparameters which are used in other files. This contains the learning rate, epochs, sigma, data directory, result directory, etc.
    4) test.py: The trained model can be evaluated with SSIM and PSNR values using this file. Make sure to change the directory in the code.
    5) loss.py: This file contains the implementation of Monte Carlo LURE and Mean Square Error loss function. Both of the functions take pytorch tensors as input and return the pytorch tensor (scalar) as output. You can just take this file and use it with your custom code as well.


NOTE:- There are no model implementations as model architecture has been taken directly from the deep inverse library.




requirements.txt: contains all the required libraries with there needed versions.
