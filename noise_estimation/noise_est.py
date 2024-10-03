import cv2
import numpy as np

import argparse

np.random.seed(42)

parser=argparse.ArgumentParser("Noise Estimation")


parser.add_argument("--img_path",default='../data/noisy_data/noisy_boat.png')

def left_shift_subtract_cols(image):
  rows, cols = image.shape
  shifted_image = np.zeros_like(image)
  shifted_image[:, 1:] = image[:, :-1]  # Shift image to the left by 1 column
  difference = - (image - shifted_image)
  return difference

def left_shift_subtract_rows(image):
  rows, cols = image.shape
  shifted_image = np.zeros_like(image)
  shifted_image[1:, :] = image[:-1, :]  # Shift image to the left by 1 column
  difference = - (image - shifted_image)
  return difference

def left_shift_subtract_rows_cols(image):
  rows, cols = image.shape
  shifted_image = np.zeros_like(image)
  shifted_image[1:, :] = image[:-1, :]  # Shift image to the left by 1 column
  difference = - (image - shifted_image)
  shifted_image1 = np.zeros_like(difference)
  shifted_image1[:, 1:] = difference[:, :-1]
  difference1=shifted_image1-difference
  return difference1



args=parser.parse_args()
img=cv2.imread(args.img_path)
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
original_gray = img_gray.copy()
m,n=img_gray.shape
img_gray= img_gray.astype(np.float32)
img_gray = (img_gray +np.ones(img_gray.shape)) * (255/256)
noisy_img= np.copy(img_gray)

diff_cols=left_shift_subtract_cols(np.log(noisy_img))
median_filtered1 = cv2.medianBlur(np.abs(diff_cols).astype(np.float32), ksize=3)
m,n=median_filtered1.shape
med1=np.median(np.abs(diff_cols))/0.6745
diff_rows=left_shift_subtract_rows(np.log(noisy_img))

median_filtered2 = cv2.medianBlur(np.abs(diff_rows).astype(np.float32), ksize=3)
med2=np.median(np.abs(diff_rows))/0.6745

diff_corner=left_shift_subtract_rows_cols(np.log(noisy_img))
med3=np.median(np.abs(diff_corner))/(0.6745*np.sqrt(2))

med=(med1+med2+med3)/(3*np.sqrt(2))
print('The final estimate of variance is : ',med)
