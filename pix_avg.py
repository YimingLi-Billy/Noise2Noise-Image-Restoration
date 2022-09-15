import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model
from math import log10, sqrt
import matplotlib.pyplot as plt
from skimage import io


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, default="",
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25,0",
                        help="noise model for test images")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image = args.image
    weight_file = args.weight_file
    val_noise_model = get_noise_model(args.test_noise_model)
    model = get_model(args.model)

    image = cv2.imread(image)
    if weight_file == "":
        #image = io.imread(image)
        print(np.mean(image))
        return 0

    model.load_weights(weight_file)
    h, w, _ = image.shape
    image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
    noise_image = val_noise_model(image)
    pred = model.predict(np.expand_dims(noise_image, 0))
    denoised_image = get_image(pred[0])

    print(np.mean(denoised_image))
    return 0


if __name__ == '__main__':
    main()