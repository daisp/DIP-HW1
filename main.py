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
        blurred_images.append(blurred_image)
        if show:
            plt.figure()
            plt.imshow(blurred_image, cmap='gray')
            plt.show()
        if i == stop:
            break
    return blurred_images


def restore_image(images, origin_image, show=True, p=np.inf):
    psnr = lambda orig_img, img2: 10 * math.log10((255 ** 2) / np.mean(np.power(orig_img - img2, 2)))
    psnr_list = []
    restored_img = None
    if p is np.inf:
        restored_img = compute_by_max(images, restored_img, psnr, psnr_list, origin_image)
    else:
        restored_img = compute_by_p(images, p, restored_img, psnr, psnr_list, origin_image)
    if show:
        plt.figure()
        plt.imshow(restored_img, cmap='gray')
        plt.show()
    return psnr_list, restored_img


def get_weighted_sum(numerators, denominator, m, images):
    weighted_sum = np.zeros_like(fft2(images[0]))
    for i in range(m):
        v_hat = fft2(images[i])
        weighted_sum += np.multiply(np.divide(numerators[i], denominator), v_hat)
    return weighted_sum


def compute_by_p(images, p, restored_img, psnr, psnr_list, origin_image):
    print("p is {}".format(p))
    sum_abs_v_hat_to_the_p_0_to_m = np.zeros_like(fft2(images[0]))
    v_hat_to_the_p_list = []
    for m in range(len(images)):
        v_hat = fft2(images[m])
        abs_v_hat_to_the_p = np.power(np.abs(v_hat), p)
        sum_abs_v_hat_to_the_p_0_to_m += abs_v_hat_to_the_p
        v_hat_to_the_p_list.append(abs_v_hat_to_the_p)
        weighted_sum = get_weighted_sum(v_hat_to_the_p_list, sum_abs_v_hat_to_the_p_0_to_m, m, images)
        restored_img = np.abs(ifft2(weighted_sum))
        psnr_list.append(psnr(origin_image, restored_img))
    return restored_img


def compute_by_max(images, restored_img, psnr, psnr_list, origin_image):
    print("p is infinity")
    fba_image = None
    for image in images:
        if fba_image is None:
            fba_image = fft2(image).copy()
        else:
            image_ft = fft2(image)
            fba_image = np.where(np.abs(image_ft) > np.abs(fba_image), image_ft, fba_image)
        restored_img = np.abs(ifft2(fba_image))
        psnr_list.append(psnr(origin_image, restored_img))
    return restored_img


def generate_psf(cont_x, cont_y):
    indxs = np.linspace(0, 1001, num=256, endpoint=False, dtype=int)
    discrete = np.array((cont_x[indxs], cont_y[indxs]))
    discrete[0] /= np.max(np.abs(discrete[0]))
    discrete[1] /= np.max(np.abs(discrete[1]))
    discrete *= (PSF_SIZE - 1) // 2
    discrete = np.round(discrete)
    psf = np.zeros((PSF_SIZE, PSF_SIZE), dtype=int)
    for x, y in zip(discrete[0], discrete[1]):
        psf[int(x) + (PSF_SIZE - 1) // 2, int(y) + (PSF_SIZE - 1) // 2] += 1
    return psf / np.sum(psf)


if __name__ == '__main__':
    original_image = cv2.imread(str(Path('./hw1/DIPSourceHW1.jpg')), 0)
    blurred_images = blur_images_and_show(original_image, show=False, stop=0)
    psnr_list_results, restored_image = restore_image(blurred_images, original_image, show=False, p=20)
    plt.figure()
    plt.plot(psnr_list_results)
    plt.figure()
    plt.imshow(original_image, cmap='gray')
    plt.figure()
    plt.imshow(restored_image, cmap='gray')
    plt.show()
