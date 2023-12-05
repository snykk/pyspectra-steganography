import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class Analysis(object):
    def psnrAnalysis(self, image1: np.ndarray, image2: np.ndarray) -> np.float64:
        """
        Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
        
        Args:
        - image1: First image for PSNR calculation
        - image2: Second image for PSNR calculation
        
        Returns:
        - PSNR value as a measure of image similarity
        
        Calculates the PSNR (Peak Signal-to-Noise Ratio) between two provided images. First, it converts both input images to float32 format. Then, it computes the squared difference between corresponding pixels in the images. Afterward, it calculates the mean squared error (MSE) from the squared differences. In case the MSE is zero (indicating identical images), it returns a message stating that the images are identical with a PSNR value of +Infinity. Otherwise, it calculates the maximum pixel value based on the image bit depth (e.g., 255 for 8-bit images). Finally, it computes the PSNR using the formula PSNR = 20 * log10(MAX) - 10 * log10(MSE), where MAX represents the maximum pixel value. The resulting PSNR value quantifies the similarity between the input images.
        """

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
    
    def ssimAnalysis(self, image1: np.ndarray, image2: np.ndarray) -> np.float64:
        """
        Calculates the Structural Similarity Index (SSIM) between two images.
        
        Args:
        - image1: First image for SSIM calculation
        - image2: Second image for SSIM calculation
        
        Returns:
        - SSIM value as a measure of image similarity
        
        Computes the Structural Similarity Index (SSIM) between two provided images. First, it converts both 
        input images to float32 format. Then, it utilizes the SSIM function from a suitable library (e.g., 
        scikit-image or OpenCV) to calculate the SSIM value. The function is invoked with specific parameters, 
        including 'full=True' to return the full SSIM image and 'data_range=1.0' to specify the range of the 
        input image data. Finally, it returns the SSIM value, which represents the similarity between the two 
        input images based on their structural information.
        """

        # Convert images to float32
        image1 = np.float32(image1)
        image2 = np.float32(image2)

        # Calculate SSIM
        ssim_value = ssim(image1, image2, full=True, data_range=1.0)[0]
        return ssim_value
    