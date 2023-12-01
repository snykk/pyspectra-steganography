import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class Analysis(object):
    def psnrAnalysis(self, image1=np.array([]), image2=np.array([])):
        # Convert images to float32
        image1 = np.float32(image1)
        image2 = np.float32(image2)

        # Calculate the squared difference between the images
        diff = image1 - image2
        squared_diff = diff ** 2

        # Calculate the mean squared error
        mse = np.mean(squared_diff)

        # To avoid division by zero, check for zero MSE
        if mse == 0:
            return "Identical images (PSNR = +Infinity)"

        # Calculate the maximum pixel value
        max_pixel = 255.0  # For 8-bit images

        # Calculate PSNR using the formula: PSNR = 20 * log10(MAX) - 10 * log10(MSE)
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
        return psnr
    
    def ssimAnalysis(self, image1=np.array([]), image2=np.array([])):
        # Convert images to float32
        image1 = np.float32(image1)
        image2 = np.float32(image2)

        # Calculate SSIM
        ssim_value = ssim(image1, image2, full=True, data_range=1.0)[0]
        return ssim_value
    