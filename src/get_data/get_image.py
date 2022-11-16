import numpy as np


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


def get_cropped_image(image, crop_coordinates, pad=0):
    h, w = image.shape[0], image.shape[1]
    x_min, y_min, x_max, y_max = crop_coordinates
    cropped_image = image[max(y_min - pad, 0):min(y_max + pad, h), max(x_min - pad, 0):min(x_max + pad, w)]
    return cropped_image


def get_image(name, old_version=False):
    img_raw = np.load(name)['raw_image_low']
    img_raw = img_raw if old_version else ((img_raw / img_raw.max()) * 255).astype(np.uint8)
    img0 = np.copy(img_raw)
    img0 = eliminate_lines(img0)
    return img0