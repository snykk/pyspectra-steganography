import cv2
from pyspectra.core import core
from pyspectra.analysis import analysis
from pyspectra.exception import pyspectra_exception

def main():
    while True:
        mode = input("embed/extract/analyze? ")
        if mode == "embed":
            image_name = input("Enter cover image: ")
            key = input("Enter the key: ")
            scalar = int(input("Enter scalar value [1-255]: "))
            message = input("Enter the message: ")
            output_name = input("Enter the name of output image: ")

            pyspectra = core.PySpectra()
            try:
                cover = cv2.imread("./assets/" + image_name, 0)
                new_image = pyspectra.embedding(image=cover, message=message, key=key, scalar=scalar)
                cv2.imwrite("./outputs/" + output_name, new_image)
            except pyspectra_exception.PySpectraException as pe:
                print("PySpectra Exception:", pe)
            except Exception as e:
                print("Error:", e)

        elif mode == "extract":
            stego_name = input("Enter stego name: ")
            key = input("Enter the key: ")

            pyspectra = core.PySpectra()
            try:
                stego = cv2.imread("./outputs/" + stego_name, 0)
                extracted_message = pyspectra.extract(image=stego, key=key)
                print("Message:", extracted_message)
            except pyspectra_exception.PySpectraException as pe:
                print("PySpectra Exception:", pe)
            except Exception as e:
                print("Error:", e)

        elif mode == "analyze":
            try:
                image1_path = "./assets/" + input("Enter cover image: ")
                image2_path = "./outputs/" + input("Enter stego image: ")

                image1 = cv2.imread(image1_path, 0)
                image2 = cv2.imread(image2_path, 0)

                analyze = analysis.Analysis()
                
                psnr = analyze.psnrAnalysis(image1, image2)
                ssim = analyze.ssimAnalysis(image1, image2)

                # output
                # PSNR: 84.91257479169302
                # SSIM: 0.9999983332798968

                print("PSNR:", psnr)
                print("SSIM:", ssim)
            except pyspectra_exception.PySpectraException as pe:
                print("PySpectra Exception:", pe)
            except Exception as e:
                print("Error:", e)
        else:
            print("Invalid mode. Please enter 'embed', 'extract', or 'analyze'.")

if __name__ == "__main__":
    main()
