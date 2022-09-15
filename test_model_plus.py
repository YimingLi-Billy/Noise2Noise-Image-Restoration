import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model
from math import log10, sqrt


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--noise_weight_file", type=str, required=True,
                        help="noise weight file")
    parser.add_argument("--noise_weight_file2", type=str, default=None,
                        help="noise weight file2")
    parser.add_argument("--clean_weight_file", type=str, required=True,
                        help="clean weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25,0",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    args = get_args()
    image_dir = args.image_dir
    noise_weight_file = args.noise_weight_file
    clean_weight_file = args.clean_weight_file
    val_noise_model = get_noise_model(args.test_noise_model)
    noise_model = get_model(args.model)
    clean_model = get_model(args.model)
    noise_model.load_weights(noise_weight_file)
    clean_model.load_weights(clean_weight_file)

    if args.noise_weight_file2:
        noise_weight_file2 = args.noise_weight_file2
        noise_model2 = get_model(args.model)
        noise_model2.load_weights(noise_weight_file2)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape
        image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
        h, w, _ = image.shape

        if args.noise_weight_file2:
            out_image = np.zeros((h, w * 5, 3), dtype=np.uint8)
        else:
            out_image = np.zeros((h, w * 4, 3), dtype=np.uint8)
        noise_image = val_noise_model(image)
        noise_pred = noise_model.predict(np.expand_dims(noise_image, 0))
        noise_denoised_image = get_image(noise_pred[0])
        clean_pred = clean_model.predict(np.expand_dims(noise_image, 0))
        clean_denoised_image = get_image(clean_pred[0])
        if args.noise_weight_file2:
            noise_pred2 = noise_model2.predict(np.expand_dims(noise_image, 0))
            noise_denoised_image2 = get_image(noise_pred2[0])
        out_image[:, :w] = image
        out_image[:, w:w * 2] = noise_image
        out_image[:, w * 2:w * 3] = noise_denoised_image
        out_image[:, w * 3:w * 4] = clean_denoised_image
        if args.noise_weight_file2:
            out_image[:, w * 4:w * 5] = noise_denoised_image2
        noise_psnr = PSNR(image, noise_denoised_image)
        clean_psnr = PSNR(image, clean_denoised_image)
        if args.noise_weight_file2:
            noise_psnr2 = PSNR(image, noise_denoised_image2)

        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
        else:
            cv2.imshow("result", out_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0
        if args.noise_weight_file2:
            print(image_path.name + '    noise_psnr: ', noise_psnr, '    clean_psnr: ', clean_psnr, '    noise_psnr2: ', noise_psnr2,'\n')
        else:
            print(image_path.name + '    noise_psnr: ', noise_psnr, '    clean_psnr: ', clean_psnr, '\n')


if __name__ == '__main__':
    main()
