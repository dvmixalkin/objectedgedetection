import numpy as np
import cv2
import json
from matplotlib import pyplot as plt


def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src


# Image preprocessing
def eliminate_horizontal_lines(img, columns_range=500):
    I0 = np.quantile(img[:, :columns_range], 0.99)
    C = np.mean(img[:, :columns_range], axis=1) / I0
    I0 = 1 if I0 == 0 else I0
    C = np.where(C == 0, 1, C)
    C = np.expand_dims(C, 1)
    img = np.clip(img / C, 0, I0)
    return img


def eliminate_vertical_lines(img, rows_range=500):
    I0 = np.quantile(img[:rows_range, :], 0.99)
    C = np.mean(img[:rows_range, :], axis=0) / I0
    I0 = 1 if I0 == 0 else I0
    C = np.where(C == 0, 1, C)
    img = np.clip(img / C, 0, I0)
    return img


def eliminate_lines(img, columns_range=500, eliminate_hotizontal_lines=True, eliminate_vert_lines=True):
    if eliminate_hotizontal_lines:
        img = eliminate_horizontal_lines(img, columns_range=columns_range)
    if eliminate_vert_lines:
        img = eliminate_vertical_lines(img, rows_range=columns_range)
    return img


def plot(image_src, image_eq, cmap_val):
    fig = plt.figure(figsize=(10, 20))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis("off")
    ax1.title.set_text('Original')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis("off")
    ax2.title.set_text("Equalized")

    ax1.imshow(image_src, cmap=cmap_val)
    ax2.imshow(image_eq, cmap=cmap_val)


def equalize_this_v1(image_file, with_plot=False, gray_scale=False):
    if isinstance(image_file, str):
        image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    else:
        image_src = image_file

    if not gray_scale:
        r_image, g_image, b_image = cv2.split(image_src)

        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)

        image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        cmap_val = None
    else:
        image_eq = cv2.equalizeHist(image_src)
        cmap_val = 'gray'

    if with_plot:
        plot(image_src, image_eq, cmap_val)
        return True
    return image_eq


def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq


def equalize_this_v2(image_file, with_plot=False, gray_scale=False):
    if isinstance(image_file, str):
        image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    else:
        image_src = image_file
    if not gray_scale:
        r_image = image_src[:, :, 0]
        g_image = image_src[:, :, 1]
        b_image = image_src[:, :, 2]

        r_image_eq = enhance_contrast(image_matrix=r_image)
        g_image_eq = enhance_contrast(image_matrix=g_image)
        b_image_eq = enhance_contrast(image_matrix=b_image)

        image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
        cmap_val = None
    else:
        image_eq = enhance_contrast(image_matrix=image_src)
        cmap_val = 'gray'

    if with_plot:
        plot(image_src, image_eq, cmap_val)
        return True
    return image_eq


def equalize_this_v1_vs_v2(image_file, with_plot=False, gray_scale=False):
    if isinstance(image_file, str):
        image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    else:
        image_src = image_file

    cmap_val = None if not gray_scale else 'gray'

    image_eq_v1 = equalize_this_v1(image_file=image_src, with_plot=False)
    image_eq_v2 = equalize_this_v2(image_file=image_src, with_plot=False)
    # diff = image_eq_v1 - image_eq_v2
    # diff_v12 = np.clip(image_eq_v1 - image_eq_v2, 0)
    # diff_v21 = image_eq_v2 - image_eq_v1
    if with_plot:
        fig = plt.figure(figsize=(15, 30))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.axis("off")
        ax2.title.set_text("Equalized_v1")

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.axis("off")
        ax3.title.set_text("Equalized_v2")

        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_eq_v1, cmap=cmap_val)
        ax3.imshow(image_eq_v2, cmap=cmap_val)


def main():
    from PIL import Image
    npz = np.load('../../../examples/input/20210331STM0104727.npz')['raw_image_low']
    image_src = (npz / npz.max()) * 255
    image_src = eliminate_lines(image_src)
    image_src = np.array(Image.fromarray(image_src).convert('RGB'))

    # equalize_this_v1(image_file=image_src, with_plot=True)
    # equalize_this_v2(image_file=image_src, with_plot=True)  # , gray_scale=True
    equalize_this_v1_vs_v2(image_src, with_plot=True, gray_scale=False)


if __name__ == '__main__':
    main()
