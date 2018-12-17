import math

import scipy.io as sio
from scipy.signal import convolve2d as conv2d
from scipy.fftpack import fft2, ifft2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

PSF_SIZE = 7


def blur_images_and_show(image, show=True, stop=0):
    mat = sio.loadmat(Path('./hw1/100_motion_paths.mat'))
    x_mat = mat['X']
    y_mat = mat['Y']
    blured_images = []
    i = 0
    for traj in zip(x_mat, y_mat):
        i += 1
        plt.figure()
        x, y = traj
        plt.plot(y, x)
        plt.gca().invert_yaxis()
        psf = generate_psf(*traj)
        plt.figure()
        plt.imshow(psf, cmap='gray')
        plt.figure()
        blurred_image = conv2d(image, psf, 'same')
        blured_images.append(blurred_image)
        plt.imshow(blurred_image, cmap='gray')
        if show:
            plt.show()
        if i == stop:
            break
    return blured_images


def restore_image(images, origin_image):
    psnr = lambda orig_img, img2: 10 * math.log10(np.amax(orig_img) ** 2) / np.mean((orig_img - img2) ** 2)
    psnr_list = []
    fba_image = None
    for image in images:
        if fba_image is None:
            fba_image = fft2(image)
        else:
            fba_image = np.maximum(np.abs(fba_image), np.abs(fft2(image)))
        ifft_ = np.abs(ifft2(fba_image))
        n = psnr(origin_image, ifft_)
        print(n)
        psnr_list.append(n)
    return psnr_list


def generate_psf(cont_x, cont_y):
    indxs = np.linspace(0, 1001, num=256, endpoint=False, dtype=int)
    discrete = np.array((cont_x[indxs], cont_y[indxs]))
    discrete = np.round(discrete)
    psf = np.zeros((PSF_SIZE, PSF_SIZE), dtype=int)
    for x, y in zip(discrete[0], discrete[1]):
        psf[int(x)+PSF_SIZE//2, int(y)+PSF_SIZE//2] += 1
    return psf


if __name__ == '__main__':
    image = cv2.imread(str(Path('./hw1/DIPSourceHW1.jpg')), 0)
    images = blur_images_and_show(image, show=False, stop=5)
    # print(len(images))
    res = restore_image(images, image)
    plt.figure()
    plt.plot(res)
    plt.show()
