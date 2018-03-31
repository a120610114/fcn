#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import time

import tensorflow as tf
import os

from utils import (_B_MEAN, _G_MEAN, _R_MEAN, _mean_image_subtraction)

# TFRcord文件
TRAIN_FILE = 'fcn_train.record'
VALIDATION_FILE = 'fcn_val.record'

# 图片信息
NUM_CLASSES = 21


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/label': tf.FixedLenFeature([], tf.string)
    })
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = tf.decode_raw(features['image/label'], tf.uint8)
    shape = tf.shape(image)
    label = tf.reshape(label, [shape[0], shape[1], 1])

    return image, label


def inputs(data_set, train=True, batch_size=1, num_epochs=1, upsample_factor_for_whole_net=32):
    assert os.path.exists(data_set), '[{0}] not exist!!!'.format(data_set)
    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input') as scope:
        filename_queue = tf.train.string_input_producer([data_set], num_epochs=num_epochs)
    image, label = read_and_decode(filename_queue)

    # Convert image to float32 before subtracting the
    # mean pixel value
    image_float = tf.to_float(image, name='ToFloat')
    # Subtract the mean pixel value from each pixel
    mean_centered_image = _mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])

    if train:
        seed = int(time.time())
        img_corped = tf.random_crop(mean_centered_image, [224, 224, 3], seed=seed)
        img_orig_corped = tf.random_crop(image, [224, 224, 3], seed=seed)
        lbl_corped = tf.random_crop(label, [224, 224, 1], seed=seed)

        images, origin_images, labels = tf.train.shuffle_batch([img_corped, img_orig_corped, lbl_corped],
                                                               batch_size=batch_size,
                                                               num_threads=4,
                                                               capacity=100 + 3 * batch_size,
                                                               min_after_dequeue=100)
    else:
        original_shape = tf.shape(image_float)[0:2]

        target_input_size_factor = tf.ceil(
            tf.div(tf.to_float(original_shape), tf.to_float(upsample_factor_for_whole_net)))
        target_input_size = tf.to_int32(tf.multiply(target_input_size_factor, upsample_factor_for_whole_net))
        padding_size = (target_input_size - original_shape) // 2

        mean_centered_image = tf.image.pad_to_bounding_box(mean_centered_image,
                                                           padding_size[0],
                                                           padding_size[1],
                                                           target_input_size[0],
                                                           target_input_size[1])

        annotation_tensor_paded = tf.image.pad_to_bounding_box(label,
                                                               padding_size[0],
                                                               padding_size[1],
                                                               target_input_size[0],
                                                               target_input_size[1])

        origin_image = tf.image.pad_to_bounding_box(image,
                                                    padding_size[0],
                                                    padding_size[1],
                                                    target_input_size[0],
                                                    target_input_size[1])
        images = tf.expand_dims(mean_centered_image, 0)
        origin_images = tf.expand_dims(origin_image, 0)
        labels = tf.expand_dims(annotation_tensor_paded, 0)

    return images, origin_images, labels
