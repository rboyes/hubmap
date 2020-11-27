import os
import glob
import numpy as np
import random
import json
import pandas as pd

from models import unet4, unet_resnet, overlap, overlap_loss
from image_utils import get_shape, basic_generator

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import argparse
from configuration import config

parser = argparse.ArgumentParser(description="Train a UNET model given input image/mask pairs",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('images_dir', help="directory where training images are")
parser.add_argument('masks_dir', help="directory where masks are")
parser.add_argument('model_dir', help="directory where model should be put")
parser.add_argument('-e', '--epochs', help='number of epochs to train for', default=config['epochs'])

cmd_args = parser.parse_args()

image_paths = sorted(glob.glob(os.path.join(cmd_args.images_dir, '*.*')))
mask_paths = sorted(glob.glob(os.path.join(cmd_args.masks_dir, '*.*')))

if len(image_paths) != len(mask_paths):
    raise ValueError("Inconsistent number of masks and images")

for image_path, mask_path in zip(image_paths, mask_paths):
    image_basename = os.path.basename(image_path)
    mask_basename = os.path.basename(mask_path)
    image_basename = os.path.splitext(image_basename)[0]
    mask_basename = os.path.splitext(mask_basename)[0]

    if image_basename != mask_basename:
        raise ValueError(f"Inconsistent image/mask pairing {image_basename} {mask_basename}")


if not os.path.exists(cmd_args.model_dir):
    os.mkdir(cmd_args.model_dir)

random.seed(7)
np.random.seed(7)

csv_logger = CSVLogger(os.path.join(cmd_args.model_dir, "log.txt"), append=True)
check_pointer = ModelCheckpoint(filepath=os.path.join(cmd_args.model_dir, "model.h5"),
                                verbose=1, save_best_only=True)

model, model_type = unet_resnet(get_shape(image_paths[0]))

train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = \
    train_test_split(image_paths, mask_paths, test_size=config['test_size'], shuffle=False)

print(model.summary())
print("number of layers = %d" % (len(model.layers)))


model.compile(optimizer=Adam(lr=config['learning_rate']), loss=overlap_loss, metrics=[overlap])

training_generator = basic_generator(train_image_paths,
                                     train_mask_paths,
                                     batch_size=config['batch_size'],
                                     imagenet_preprocess=model_type)

validation_generator = basic_generator(test_image_paths,
                                       test_mask_paths,
                                       batch_size=config['batch_size'],
                                       imagenet_preprocess=model_type)

with open(os.path.join(cmd_args.model_dir, 'config.json'), 'w') as file:
    file.write(json.dumps(config))
df_testpaths = pd.DataFrame({'test_images': test_image_paths, 'test_masks': test_mask_paths})
df_testpaths.to_csv(os.path.join(cmd_args.model_dir, "test_images.csv"), index=False)

model.fit_generator(training_generator,
                    steps_per_epoch=(len(train_image_paths) // config['batch_size']),
                    epochs=cmd_args.epochs,
                    verbose=1,
                    callbacks=[check_pointer, csv_logger],
                    validation_data=validation_generator,
                    validation_steps=(len(test_image_paths) // config['batch_size']))
