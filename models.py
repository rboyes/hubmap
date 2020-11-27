from tensorflow.keras.layers import (concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,
                                     BatchNormalization, Activation, Input, SpatialDropout2D, UpSampling2D)
from tensorflow.keras.models import Model

import tensorflow.keras.backend as keras_backend

from tensorflow.keras.applications import resnet50

import tensorflow as tf

def min_overlap(ytrue, ypred):
    """
    Calculates a fuzzy overlap
    :param ytrue: true membership of data
    :param ypred: predicted membership of data
    :return: overlap value between ytrue and ypred
    """
    ytrue_flat = keras_backend.flatten(ytrue)
    ypred_flat = keras_backend.flatten(ypred)
    intersection = keras_backend.minimum(ytrue_flat, ypred_flat)
    union = keras_backend.maximum(ytrue_flat, ypred_flat)
    overlap = keras_backend.sum(intersection) / keras_backend.sum(union)
    return overlap

def overlap(ytrue, ypred):
    """
    Calculates a fuzzy overlap
    :param ytrue: true membership of data
    :param ypred: predicted membership of data
    :return: overlap value between ytrue and ypred
    """
    _epsilon = 1.0E-6
    intersections = tf.reduce_sum(ytrue*ypred)
    unions = tf.reduce_sum(ytrue + ypred)
    score = (2.0*intersections + _epsilon)/(unions + _epsilon)
    return score

def probabilistic_overlap(ytrue, ypred):
    ytrue_flat = keras_backend.flatten(ytrue)
    ypred_flat = keras_backend.flatten(ypred)
    intersection = ytrue_flat * ypred_flat
    union = (ytrue_flat + ypred_flat) - intersection
    overlap = (keras_backend.sum(intersection)) / (keras_backend.sum(union))
    return overlap


def overlap_loss(ytrue, ypred):
    """
    cost function for an overlap measurement
    :param ytrue:
    :param ypred:
    :return:
    """
    return 1.0 - overlap(ytrue, ypred)


def _conv3_block(inputs, size):
    convolved = Conv2D(size, (3, 3), padding='same')(inputs)
    convolved = BatchNormalization(axis=-1)(convolved)
    convolved = Activation('relu')(convolved)

    convolved = Conv2D(size, (3, 3), padding='same')(convolved)
    convolved = BatchNormalization(axis=-1)(convolved)
    convolved = Activation('relu')(convolved)

    convolved = Conv2D(size, (3, 3), padding='same')(convolved)
    convolved = BatchNormalization(axis=-1)(convolved)
    convolved = Activation('relu')(convolved)
    return convolved


def _conv_block(inputs, size):
    convolved = Conv2D(size, (3, 3), padding='same')(inputs)
    convolved = BatchNormalization(axis=-1)(convolved)
    convolved = Activation('relu')(convolved)

    convolved = Conv2D(size, (3, 3), padding='same')(convolved)
    convolved = BatchNormalization(axis=-1)(convolved)
    convolved = Activation('relu')(convolved)

    return convolved


def _merge_block(lores, hires, size):
    up = Conv2DTranspose(size, (2, 2), strides=(2, 2), padding='same')(lores)
    merged = concatenate([up, hires])
    conv = _conv_block(merged, size)
    return conv


def _pool_block(inputs, size):
    conv = _conv_block(inputs, size)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def _end_block(x, spatial_dropout=0.0):
    if spatial_dropout > 0.0:
        x = SpatialDropout2D(spatial_dropout)(x)
    x = Conv2D(1, (1, 1))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('sigmoid')(x)

    return x


def unet4(input_shape, size=32):
    inputs = Input((input_shape[0], input_shape[1], input_shape[2]))

    filter_sizes = [size * 2 ** i for i in range(0, 5)]

    (conv0down, pool0down) = _pool_block(inputs, filter_sizes[0])
    (conv1down, pool1down) = _pool_block(pool0down, filter_sizes[1])
    (conv2down, pool2down) = _pool_block(pool1down, filter_sizes[2])
    (conv3down, pool3down) = _pool_block(pool2down, filter_sizes[3])
    (conv4down, pool4down) = _pool_block(pool3down, filter_sizes[4])

    centre = _conv3_block(pool4down, filter_sizes[4])

    conv4up = _merge_block(centre, conv4down, filter_sizes[4])
    conv3up = _merge_block(conv4up, conv3down, filter_sizes[3])
    conv2up = _merge_block(conv3up, conv2down, filter_sizes[2])
    conv1up = _merge_block(conv2up, conv1down, filter_sizes[1])
    conv0up = _merge_block(conv1up, conv0down, filter_sizes[0])

    end = _end_block(conv0up)

    model4u = Model(inputs=inputs, outputs=end)

    return model4u, "unet"


def get_model_type(model):
    if model.layers[7].name == 'res2a_branch2a':
        return "resnet"
    elif len(model.layers) > 200 and model.layers[260] == 'block35_10_ac':
        return "inception_resnet"
    elif "mobilenetv2" in model.name:
        return "mobilenetv2"
    else:
        return "unet"

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):

    conv = Conv2D(filters, (3, 3),
                  padding="same",
                  kernel_initializer="he_normal",
                  strides=strides,
                  name=prefix + "_conv")(prevlayer)

    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def unet_resnet(input_shape):
    resnet_base = resnet50.ResNet50(input_shape=input_shape, include_top=False)

    for layer in resnet_base.layers:
        layer.trainable = True

    conv1 = resnet_base.get_layer("activation").output
    conv2 = resnet_base.get_layer("activation_9").output
    conv3 = resnet_base.get_layer("activation_21").output
    conv4 = resnet_base.get_layer("activation_39").output
    conv5 = resnet_base.get_layer("activation_48").output

    conv6 = _merge_block(conv5, conv4, 256)
    conv7 = _merge_block(conv6, conv3, 192)
    conv8 = _merge_block(conv7, conv2, 128)
    conv9 = _merge_block(conv8, conv1, 64)

    conv10 = _merge_block(conv9, resnet_base.input, 32)
    x = _end_block(conv10, spatial_dropout=0.25)
    model = Model(resnet_base.input, x)
    return model, "resnet"
