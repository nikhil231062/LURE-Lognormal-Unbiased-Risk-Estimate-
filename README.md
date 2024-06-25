DATA:
    AFHQ: Download AFHQ dataset and place the train data in data/afhq/train/imgs folder. combine images in all the classes into imgs folder. Similarly place test data in data/afhq/val/imgs.
    Noisy Data: Contains noisy images corresponding to sigma of 0.2. Keep your noisy image in this folder inorder to estimate noise.

NOISE ESTIMATION : 
    Run noise_est.py files after replacing the path of the required noisy image. This uses median based approach for estimating the sigma of gaussian noise. Hence we take log of the image containing lognormal noise to convert noise to gaussian domain.