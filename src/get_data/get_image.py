import numpy as np
import cv2


# Image preprocessing
def eliminate_lines(img, columns_range=500, eliminate_vert_lines=True, rows_range=100):
    I0 = np.quantile(img[:, :columns_range], 0.99)
    C = np.mean(img[:, :columns_range], axis=1) / I0
    I0 = 1 if I0 == 0 else I0
    C = np.where(C == 0, 1, C)
    C = np.expand_dims(C, 1)
    img = np.clip(img / C, 0, I0)

    if eliminate_vert_lines:
        I0 = np.quantile(img[:rows_range, :], 0.99)
        C = np.mean(img[:rows_range, :], axis=0) / I0
        I0 = 1 if I0 == 0 else I0
        C = np.where(C == 0, 1, C)
        img = np.clip(img / C, 0, I0)
    return img


def get_cropped_image(image, crop_coordinates, pad=[0, 0, 0, 0]):
    h, w = image.shape[0], image.shape[1]
    x_min, y_min, x_max, y_max = crop_coordinates
    cropped_image = image[max(y_min - pad[1], 0):min(y_max + pad[3], h), max(x_min - pad[0], 0):min(x_max + pad[2], w)]
    return cropped_image


def get_image(name, old_version=False):
    try:
        img_raw = np.load(name)['raw_image_low']
    except:
        img_raw = name['raw_image_low']
    img_raw = img_raw if old_version else ((img_raw / img_raw.max()) * 255).astype(np.uint8)
    img0 = np.copy(img_raw)
    img0 = eliminate_lines(img0)
    img0 = equalize_this_v2(img0.astype(np.uint8), gray_scale=True)

    return img0


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


def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src


def equalize_this_v2(image_file, gray_scale=False):
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

    return image_eq