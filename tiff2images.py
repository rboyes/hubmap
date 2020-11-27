import numpy as np
import pandas as pd
from image_utils import rle2mask, is_empty, is_lowcontrast
import skimage.io
from skimage.transform import resize
import os
import itertools

import argparse
from configuration import config

parser = argparse.ArgumentParser(description="Convert tiff files and RLE runs to smaller image/mask pairs",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("train_path", help="path to input training file", default='train.csv')
parser.add_argument('images_dir', help="directory where images should be put")
parser.add_argument('masks_dir', help="directory where masks should be put")
parser.add_argument('-p', '--patch_size', help="size of patch", nargs=2, default=config['patch_size'], type=int)
parser.add_argument('-d', '--downsampled_size',
                    help="size of downsampled patch, ie what size will the output images be?",
                    nargs=2, default=config['model_size'], type=int)
parser.add_argument('-s', '--step_size',
                    help="step size of sampling window",
                    default=config['step_size'], type=int)

cmd_args = parser.parse_args()

df_train = pd.read_csv(cmd_args.train_path)

image_ids = df_train['id'].values.tolist()
image_paths = [os.path.join('train', f"{x}.tiff") for x in df_train['id'].values]
mask_rles = df_train['encoding'].values.tolist()

n0 = cmd_args.patch_size[0]
n1 = cmd_args.patch_size[1]
r0 = cmd_args.downsampled_size[0]
r1 = cmd_args.downsampled_size[1]
step = cmd_args.step_size

if not os.path.exists(cmd_args.images_dir):
    os.mkdir(cmd_args.images_dir)
if not os.path.exists(cmd_args.masks_dir):
    os.mkdir(cmd_args.masks_dir)

for image_id, image_path, rle_runs in zip(image_ids, image_paths, mask_rles):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist")

    image = np.squeeze(skimage.io.imread(image_path))

    mask = rle2mask(rle_runs, (image.shape[1], image.shape[0]), mask_value=255)

    range0 = np.arange(0, image.shape[0] - n0, step)
    range1 = np.arange(0, image.shape[1] - n1, step)

    starts0 = np.arange(0, image.shape[0] - n0, step)
    if starts0[-1] < (image.shape[0] - n0):
        starts0 = np.append(starts0, image.shape[0] - n0)

    starts1 = np.arange(0, image.shape[1] - n1, step)
    if starts1[-1] < (image.shape[1] - n1):
        starts1 = np.append(starts1, image.shape[1] - n1)

    starts = [start for start in itertools.product(starts0, starts1)]

    for patch_index, start in enumerate(starts):

        end = start[0] + n0, start[1] + n1

        patch_image = image[start[0]:end[0], start[1]:end[1], :]
        patch_mask = mask[start[0]:end[0], start[1]:end[1]]

        if (is_lowcontrast(patch_image, 12) or is_empty(patch_mask)) and np.random.random() < 0.9:
            print(f"Skipping patch {patch_index} at {start[0]},{start[1]} for image {image_id}...")
            continue
        else:
            print(f"Processing patch {patch_index} at {start[0]},{start[1]} for image {image_id}...")

        resampled_image = resize(patch_image, (r0, r1), anti_aliasing=config['image_antialiasing'],
                                 order=3, preserve_range=True)
        resampled_image = resampled_image.astype(np.uint8)
        patch_image_path = os.path.join(cmd_args.images_dir, "%s_%04d_%04d.png" % (image_id, start[0], start[1]))
        skimage.io.imsave(patch_image_path, resampled_image)

        resampled_mask = resize(patch_mask, (r0, r1), anti_aliasing=False, order=3, preserve_range=True)
        resampled_mask = resampled_mask.astype(np.uint8)
        patch_mask_path = os.path.join(cmd_args.masks_dir, "%s_%04d_%04d.png" % (image_id, start[0], start[1]))
        skimage.io.imsave(patch_mask_path, resampled_mask)
