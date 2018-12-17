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
    # print(image)
    mat = sio.loadmat(Path('./hw1/100_motion_paths.mat'))
    x_mat = mat['X']
    y_mat = mat['Y']
    blurred_images = []
    i = 0
    for traj in zip(x_mat, y_mat):
        i += 1
        x, y = traj
        if show:
            plt.figure()
            plt.plot(y, x)
            plt.gca().invert_yaxis()
        psf = generate_psf(*traj)
        if show:
            plt.figure()
            plt.imshow(psf, cmap='gray')
        blurred_image = conv2d(image, psf, 'same')
        # print(blurred_image)
        blurred_images.append(blurred_image)
        if show:
            plt.figure()
            plt.imshow(blurred_image, cmap='gray')
            plt.show()
        if i == stop:
            break
    return blurred_images


def restore_image(images, origin_image, show=True):
    psnr = lambda orig_img, img2: 10 * math.log10((255 ** 2) / np.mean((orig_img - img2) ** 2))
    psnr_list = []
    fba_image = None
    for image in images:
        if fba_image is None:
            fba_image = fft2(image)
        else:
            image_ft = fft2(image)
            for row in range(256):
                for col in range(256):
                    if np.abs(fba_image[row, col]) < np.abs(image_ft[row, col]):
                        fba_image[row, col] = image_ft[row, col]
        restored_image = np.abs(ifft2(fba_image))
        if show:
            plt.figure()
            plt.imshow(restored_image, cmap='gray')
            plt.show()
        n = psnr(origin_image, restored_image)
        psnr_list.append(n)
    return psnr_list, restored_image


def generate_psf(cont_x, cont_y):
    indxs = np.linspace(0, 1001, num=256, endpoint=False, dtype=int)
    discrete = np.array((cont_x[indxs], cont_y[indxs]))
    discrete[0] /= np.max(np.abs(discrete[0]))
    discrete[1] /= np.max(np.abs(discrete[1]))
    discrete *= (PSF_SIZE - 1) // 2
    discrete = np.round(discrete)
    # print(discrete[0], discrete[1])
    psf = np.zeros((PSF_SIZE, PSF_SIZE), dtype=int)
    for x, y in zip(discrete[0], discrete[1]):
        psf[int(x) + (PSF_SIZE - 1) // 2, int(y) + (PSF_SIZE - 1) // 2] += 1
    return psf / np.sum(psf)


if __name__ == '__main__':
    original_image = cv2.imread(str(Path('./hw1/DIPSourceHW1.jpg')), 0)
    blurred_images = blur_images_and_show(original_image, show=False, stop=0)
    psnr_list, restored_image = restore_image(blurred_images, original_image, show=False)
    plt.figure()
    plt.plot(psnr_list)
    plt.figure()
    plt.imshow(original_image, cmap='gray')
    plt.figure()
    plt.imshow(restored_image, cmap='gray')
    plt.show()
