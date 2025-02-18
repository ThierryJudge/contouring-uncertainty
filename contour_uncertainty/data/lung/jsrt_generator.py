import os
from typing import List

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from vital.data.config import Subset, Tags

from contour_uncertainty.data.config import ContourTags
from contour_uncertainty.data.lung.config import RLUNG, HEART, LLUNG, img_save_options, seg_save_options
from contour_uncertainty.data.lung.utils import lung_contour_to_mask

TRAIN_TXT = 'hybridnet-jsrt/train_files.txt'
VAL_TXT = 'hybridnet-jsrt/val_files.txt'
TEST_TXT = 'hybridnet-jsrt/test_files.txt'

TRAIN_POINTS = 'hybridnet-jsrt/Train/landmarks'
VAL_POINTS = 'hybridnet-jsrt/Val/landmarks'
TEST_POINTS = 'hybridnet-jsrt/Test/landmarks'

IMAGES_DIR = 'hybridnet-jsrt/All247images/'

original_img_shape = (2048, 2048)  # matrix size


def resize_image_and_contours(image: np.ndarray, contours: np.ndarray, output_shape):
    """ Resize the image and corresponding contour.

    Args:
        image: ndarray (H, W)
        contours: dict containing contours of shape (N_images, N_points, 2)
        output_shape: size of the resized image

    Returns:
        image (ndarray), contour (dict)
    """

    original_shape = image.shape[-2:]
    image = np.array(Image.fromarray(image).resize(output_shape))

    contours[..., 1] = contours[..., 1] * output_shape[0] / original_shape[0]
    contours[..., 0] = contours[..., 0] * output_shape[1] / original_shape[1]

    return image, contours


def write_set(dataset: h5py.File, image_ids: List, landmark_path: str):
    for file in tqdm(image_ids):
        dtype = np.dtype('>u2')  # big-endian unsigned integer (16bit)
        img_file = IMAGES_DIR + file + '.IMG'
        if os.path.exists(img_file):
            # Reading.
            fid = open(img_file, 'rb')
            img = np.fromfile(fid, dtype).reshape(original_img_shape)
            img = 1 - img.astype('float') / 4096
            img = img * 255
            img = np.array(Image.fromarray(img).resize((1024, 1024)))

            landmarks = np.load(f'{landmark_path}/{file}.npy').astype('float').reshape(-1, 2)
            img, landmarks = resize_image_and_contours(img, landmarks, (256, 256))

            landmarks = landmarks[:RLUNG + LLUNG + HEART]

            gt = lung_contour_to_mask(landmarks)

            # rl, ll, h = split_landmarks(landmarks)
            # f, (ax1, ax2) = plt.subplots(1, 2)
            # ax1.imshow(img, cmap='gray')
            # ax2.imshow(gt)
            # ax1.scatter(rl[:, 0], rl[:, 1], s=5)
            # ax1.scatter(rl[RL_SPECIAL, 0], rl[RL_SPECIAL, 1], marker='*')
            # ax1.scatter(ll[:, 0], ll[:, 1], s=5)
            # ax1.scatter(ll[LL_SPECIAL, 0], ll[LL_SPECIAL, 1], marker='*')
            # ax1.scatter(h[:, 0], h[:, 1], s=5)
            # ax1.scatter(h[H_SPECIAL, 0], h[H_SPECIAL, 1], marker='*')
            #
            # plt.show()

            group = dataset.create_group(file)

            group.create_dataset(name=Tags.img, data=img, **img_save_options)
            group.create_dataset(name=Tags.gt, data=gt, **seg_save_options)
            group.create_dataset(name=ContourTags.contour, data=landmarks)


with open(TRAIN_TXT) as file:
    train_files = [line.rstrip().replace('.IMG', '') for line in file]

with open(VAL_TXT) as file:
    val_files = [line.rstrip().replace('.IMG', '') for line in file]

with open(TEST_TXT) as file:
    test_files = [line.rstrip().replace('.IMG', '') for line in file]

with h5py.File('jsrt_contour.h5', "w") as dataset:
    write_set(dataset.create_group(Subset.TRAIN), train_files, TRAIN_POINTS)
    write_set(dataset.create_group(Subset.VAL), val_files, VAL_POINTS)
    write_set(dataset.create_group(Subset.TEST), test_files, TEST_POINTS)
