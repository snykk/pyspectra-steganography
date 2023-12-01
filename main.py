import cv2
from pyspectra.core import core
from pyspectra.analysis import analysis

def main():
    while True:
        mode = input("embed/extract/analyze? ")
        if mode == "embed":
            image_name = input("Masukkan nama gambar: ")
            key = input("Masukkan kunci: ")
            scalar = int(input("Masukkan nilai scalar: "))
            message = input("Masukkan pesan: ")
            output_name = input("Masukkan nama gambar output: ")

            pyspectra = core.PySpectra()
            try:
                cover = cv2.imread("./assets/" + image_name, 0)
                new_image = pyspectra.embedding(image=cover, message=message, key=key, scalar=scalar)
                cv2.imwrite("./outputs/" + output_name, new_image)
            except Exception as e:
                print("Error:", e)
                continue

        elif mode == "extract":
            stego_name = input("Masukkan nama stego: ")
            key = input("Masukkan kunci: ")

            pyspectra = core.PySpectra()
            try:
                stego = cv2.imread("./outputs/" + stego_name, 0)
                extracted_message = pyspectra.extract(image=stego, key=key)
                print("pesan:", extracted_message)
            except Exception as e:
                print("Error:", e)
                continue

        elif mode == "analyze":
            try:
                image1_path = "./assets/" + input("Masukkan path image 1: ")
                image2_path = "./outputs/" + input("Masukkan path image 2: ")

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
            except Exception as e:
                print("Error:", e)
                continue
        else:
            print("Invalid mode. Please enter 'embed', 'extract', or 'analyze'.")
            continue

if __name__ == "__main__":
    main()
