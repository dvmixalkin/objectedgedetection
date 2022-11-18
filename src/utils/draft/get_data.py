import json

import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def read_npz(path):
    try:
        npz_data = np.load(path)['raw_image_low']
        np_data = ((npz_data / npz_data.max()) * 255).astype(np.uint8)
        return np_data
    except:
        return None


def read_json(path, frmt='cascade_yolo_inference'):
    try:
        with open(path, 'r') as stream:
            json_data = json.load(stream)
        return json_data
    except:
        return None


def main():
    data_path = '../../../examples/'
    npz_folder = Path(data_path, 'input')
    for path in tqdm(os.listdir(npz_folder)):
        npz_path = str(Path(npz_folder, path))
        assert read_npz(npz_path) is not None, 'Error while image reading'
        json_path = npz_path.replace('input', 'output').replace('npz', 'json')
        assert read_json(json_path) is not None, 'Error while image reading'
    print('Done')


if __name__ == '__main__':
    main()
