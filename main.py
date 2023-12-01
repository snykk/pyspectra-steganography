import cv2
import numpy as np
from pyspectra.core import core
from pyspectra.analysis import analysis

def main():
    mode = input("embed/extract/analyze? ")
    match mode:
        case "embed":
            # image_name = input("Masukkan nama gambar: ")
            # key = input("Masukkan kunci: ")
            # scalar = int(input("Masukkan nilai scalar: "))
            # message = input("Masukkan pesan: ")
            image_name = "lenna.bmp"
            key = "lana"
            scalar = 4
            message = "coba"

            pyspectra = core.PySpectra()
            lenna = cv2.imread("./assets/" + image_name, 0)

            new_image = pyspectra.embedding(image=lenna, message=message, key=key, scalar=scalar)

            cv2.imwrite("./outputs/stego_image.bmp", new_image)

        case "extract":
            # stego_name = input("Masukkan nama stego: ")
            # key = input("Masukkan kunci: ")
            # scalar = int(input("Masukkan scalar: "))
            stego_name = "stego_image.bmp"
            key = "lana"
            scalar = 4

            pyspectra = core.PySpectra()
            stego = cv2.imread("./outputs/" + stego_name, 0)

            extracted_message = pyspectra.extract(image=stego, key=key, scalar=scalar)

            print("pesan:", extracted_message)

        case "analyze":
            image1_path = "./assets/lenna.bmp"
            image2_path = "./outputs/stego_image.bmp"

            image1 = cv2.imread(image1_path, 0)
            image2 = cv2.imread(image2_path, 0)

            analyze = analysis.Analysis()
            
            psnr = analyze.psnrAnalysis(image1, image2)
            ssim = analyze.ssimAnalysis(image1, image2)

            print("PSNR:", psnr)
            print("SSIM:", ssim)

            # output
            # PSNR: 84.91257479169302
            # SSIM: 0.9999983332798968

if __name__ == "__main__":
    main()