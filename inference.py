import numpy as np
import pandas as pd
from image_utils import mask2rle, preprocess_image
from models import overlap, overlap_loss, get_model_type
import skimage.io
from skimage.transform import resize
import os
import itertools
import json
import argparse
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description="Infer segmentation and write out to submission",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("model_dir", help="directory where model is located, after running training", required=True)
parser.add_argument("output_csv", help="Output csv file", required=True)
parser.add_argument("-s", "--submission", help="Submission template file",
                    default="sample_submission.csv")
parser.add_argument("-t", "--test_dir", help="Directory where the test tiff files are located",
                    default="test")
parser.add_argument("-m", "--masks_dir", help="directory where to write out inferred masks to", default=None)

cmd_args = parser.parse_args()

df_submission = pd.read_csv(cmd_args.submission)

image_ids = df_submission['id'].values.tolist()
image_paths = [os.path.join(cmd_args.test_dir, f"{x}.tiff") for x in df_submission['id'].values]

if not os.path.exists(os.path.join(cmd_args.model_dir), 'config.json'):
    raise FileNotFoundError(f"Error - unable to find config.json in {cmd_args.model_dir}")

with open(os.path.join(cmd_args.model_dir, 'config.json'), 'r') as file:
    config = json.load(file)
    n0 = config['patch_size'][0]
    n1 = config['patch_size'][1]
    step = config['step_size']

model = load_model(os.path.join(cmd_args.model_dir, 'model.h5'),
                   custom_objects={'overlap_loss': overlap_loss,
                                   'overlap': overlap})
model_type = get_model_type(model)

model_size = model.layers[0].input_shape[0][1:4]
r0 = model_size[0]
r1 = model_size[1]

rle_runs = []
for image_id, image_path in zip(image_ids, image_paths):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist")

    image = np.squeeze(skimage.io.imread(image_path))

    mask_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    count_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

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
        print(f"Processing patch {patch_index} of {len(starts)} for image {image_id}...")

        end = start[0] + n0, start[1] + n1

        # Extract a patch image
        patch_image = image[start[0]:end[0], start[1]:end[1], :]

        patch_image = preprocess_image(patch_image, model_type)

        downsampled_patch = resize(patch_image, (r0, r1), anti_aliasing=config['image_antialiasing'],
                                   order=3, preserve_range=True)

        downsampled_mask = model.predict(downsampled_patch[np.newaxis, :], batch_size=1)

        downsampled_mask = np.squeeze(downsampled_mask)

        # Upsample the predicted mask to match the original patch
        patch_mask = resize(downsampled_mask, (n0, n1), anti_aliasing=False, order=3, preserve_range=True)

        # Plonk that back into the super mask, and update the count
        mask_image[start[0]:end[0], start[1]:end[1]] += patch_mask
        count_image[start[0]:end[0], start[1]:end[1]] += 1.0

    mask_image = mask_image / count_image

    mask_image = np.where(mask_image > 0.5, 1, 0).astype(np.uint8)

    if cmd_args.masks_dir and os.path.exists(cmd_args.masks_dir) and os.path.isdir(cmd_args.masks_dir):
        skimage.io.imsave(os.path.join(cmd_args.masks_dir, f"{image_id}.png"))

    rle_runs.append(mask2rle(mask_image))

df_submission['id'] = image_ids
df_submission['predicted'] = rle_runs

df_submission.to_csv(cmd_args.output_csv)
