import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model
from math import log10, sqrt
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file1", type=str, required=True,
                        help="trained weight file1")
    parser.add_argument("--weight_file2", type=str,
                        help="trained weight file2")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25,0",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file1 = args.weight_file1
    model1 = get_model(args.model)
    model1.load_weights(weight_file1)
    if args.weight_file2:
        weight_file2 = args.weight_file2
        model2 = get_model(args.model)
        model2.load_weights(weight_file2)
        psnr2 = []

    
    psnr1 = []
    noise_level = np.arange(30)

    image_paths = list(Path(image_dir).glob("*.*"))


    def gaussian_noise(img, noise_level):
                noise_img = img.astype(np.float)
                noise = np.random.randn(*img.shape) * noise_level
                noise_img += noise
                noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
                return noise_img
            
    def PSNR(original, compressed): 
        mse = np.mean((original - compressed) ** 2) 
        if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                    # Therefore PSNR have no importance. 
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse)) 
        return psnr 
            
    for noise in noise_level:
        sub_psnr1 = []
        if args.weight_file2:
            sub_psnr2 = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            h, w, _ = image.shape
            image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            h, w, _ = image.shape

            noise_image = gaussian_noise(image, noise)
            pred1 = model1.predict(np.expand_dims(noise_image, 0))
            denoised_image1 = get_image(pred1[0])
            if args.weight_file2:
                pred2 = model2.predict(np.expand_dims(noise_image, 0))
                denoised_image2 = get_image(pred2[0])
                sub_psnr2.append(PSNR(image, denoised_image2))
            sub_psnr1.append(PSNR(image, denoised_image1))
        psnr1.append(np.mean(sub_psnr1))
        if args.weight_file2:
            psnr2.append(np.mean(sub_psnr2))

    plt.plot(noise_level, psnr1, label="Clean target")
    plt.plot(noise_level, psnr2, label="Noise target")
    plt.legend()
    plt.xlabel('noise_level (Gaussian std)')
    plt.ylabel('PSNR')
    plt.title('PSNR vs Noise')
    plt.show()


if __name__ == '__main__':
    main()
