import math

import scipy.io as sio
from scipy.signal import convolve2d as conv2d
from scipy.fftpack import fft2, fftshift
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

PSF_SIZE = 32

def main():
    image = cv2.imread(str(Path('./hw1/DIPSourceHW1.jpg')), 0)
    mat = sio.loadmat(Path('./hw1/100_motion_paths.mat'))
    x_mat = mat['X']
    y_mat = mat['Y']
    i = 0
    for traj in zip(x_mat, y_mat):
        i+=1
        plt.figure()
        x, y = traj
        plt.plot(y, x)
        plt.gca().invert_yaxis()
        plt.show()
        psf = generate_psf(*traj)
        plt.figure()
        plt.imshow(psf, cmap='gray')
        plt.show()
        plt.figure()
        plt.imshow(conv2d(image, psf, 'same'), cmap='gray')
        plt.show()
        if i == 1:
            break


def restore_image(images, origin_image):
    psnr = lambda orig_img, img2: 10 * math.log10(max(orig_img) ** 2) / np.mean((orig_img - img2) ** 2)
    for k in range(100):
        pass


def generate_psf(cont_x, cont_y):
    indxs = np.linspace(0, 1001, num=256, endpoint=False, dtype=int)
    discrete = np.array((cont_x[indxs], cont_y[indxs]))
    discrete = np.round(discrete / np.max(np.abs(discrete)) * (PSF_SIZE - 1)//2)
    psf = np.zeros((PSF_SIZE, PSF_SIZE), dtype=int)
    for x, y in zip(discrete[0], discrete[1]):
        psf[int(x)+PSF_SIZE//2, int(y)+PSF_SIZE//2] += 1
    return psf


if __name__ == '__main__':
    main()
