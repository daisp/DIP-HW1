import math

import scipy.io as sio
from scipy.signal import convolve2d as conv2d
from scipy.fftpack import fft2, fftshift
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    image = cv2.imread(str(Path('./hw1/DIPSourceHW1.jpg')), 0)
    mat = sio.loadmat(Path('./hw1/100_motion_paths.mat'))
    x_mat = mat['X']
    y_mat = mat['Y']
    i=0
    for traj in zip(x_mat, y_mat):
        plt.figure()
        plt.plot(*traj)
        plt.show()
        psf = generate_psf(*traj)
        plt.figure()
        plt.imshow(psf, cmap='gray')
        plt.show()
        plt.figure()
        plt.imshow(conv2d(image, psf, 'same'), cmap='gray')
        plt.show()
        break


def restore_image(images, origin_image):
    psnr = lambda orig_img, img2: 10 * math.log10(max(orig_img) ** 2) / np.mean((orig_img - img2) ** 2)
    for k in range(100):
        pass


def generate_psf(cont_x, cont_y):
    discrete_x = cont_x[0::3]
    discrete_x += -min(discrete_x)
    discrete_x /= max(discrete_x)
    discrete_y = cont_y[0::3]
    discrete_y += -min(discrete_y)
    discrete_y /= max(discrete_y)
    psf = np.zeros((discrete_y.shape[0], discrete_x.shape[0]))
    for xx, yy in zip(discrete_x, discrete_y):
        psf[int(round(yy * 255)), int(round(xx * 255))] = 1
    return psf / psf.sum()


if __name__ == '__main__':
    main()
