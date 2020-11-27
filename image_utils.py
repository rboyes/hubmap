import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


def is_lowcontrast(image, threshold=25):
    """
    :param image: a numpy based image, 3 channels, range 0:255
    :param threshold: a threshold on the variance for each channel
    :return: True if the variance in all channels is below the threshold, analagous to a low contrast image
    """
    for channel in range(3):
        if np.var(image[:, :, channel]) > threshold:
            return False

    return True


def is_empty(mask):
    """
    :param mask: a numpy based mask, single channel
    :return: True if all the pixels are zero, ie an empty mask
    """
    if np.max(mask) == 0:
        return True
    else:
        return False


# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600, 256), mask_value=1):
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = mask_value
    return img.reshape(shape).T


def get_shape(image_path):
    """
    :param image_path: path to an image file
    :return: returns the dimensions of the image
    """
    test_image = img_to_array(load_img(image_path))

    if len(test_image.shape) == 2:
        return test_image.shape[0], test_image.shape[1]
    elif len(test_image.shape) == 3:
        return test_image.shape[0], test_image.shape[1], test_image.shape[2]
    raise ValueError("Unknown number of image dimensions")


def preprocess_image(image, model_type="resnet"):
    if model_type == "resnet":
        return preprocess_input(image)
    elif model_type == "inception_resnet":
        raise NotImplemented("Have not implemented preprocessing for inception-resnet yet")
    else:
        return image / 255.0


def augment_image(image, mask):
    import random
    if random.choice([False, True]):
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    rot = np.random.choice([0, 1, 2, 3])

    if rot > 0:
        image = np.rot90(image, rot)
        mask = np.rot90(mask, rot)

    return image, mask

def basic_generator(image_paths, mask_paths, batch_size=16, imagenet_preprocess="resnet"):
    while True:

        np_image_paths = np.array(image_paths)
        np_mask_paths = np.array(mask_paths)

        n0, n1, nchannel = get_shape(image_paths[0])

        num_images = len(image_paths)
        image_index = np.random.permutation(num_images)

        for index in range(0, num_images, batch_size):
            end_index = min(index + batch_size, num_images)

            index_bs = end_index - index
            batch_image_paths = np_image_paths[image_index[index:end_index]].tolist()
            batch_mask_paths = np_mask_paths[image_index[index:end_index]].tolist()
            images = np.ndarray((index_bs, n0, n1, nchannel), dtype=np.float32)
            masks = np.ndarray((index_bs, n0, n1, 1), dtype=np.float32)

            for batch_index in range(index_bs):
                image = img_to_array(load_img(batch_image_paths[batch_index]))
                mask = img_to_array(load_img(batch_mask_paths[batch_index], color_mode='grayscale'))

                image, mask = augment_image(image, mask)

                images[batch_index] = image
                masks[batch_index] = mask / 255.0

            images = preprocess_image(images, imagenet_preprocess)

            yield images, masks
