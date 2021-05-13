#   -*- coding: utf-8 -*-
#  Copyright (c)  2021.  Jeffrey Nirschl. All rights reserved.
#
#  Licensed under the MIT license. See the LICENSE file in the project
#  root directory for  license information.
#
#  Time-stamp: <>
#   ======================================================================

import os
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D


def conv_relu_bn(filters, kernel_size=(3,3), strides=(1, 1), padding="valid"):
    return [tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=strides, padding=padding,
                                   activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            ]


def simple_mnist(base_filter=32, fc_width=512, dropout_rate=0.5,
                 image_size=(28,28,1), n_class=10, learn_rate=0.01,
                 optimizer="adam"):
    """Simple CNN implementation for MNIST"""
    assert (base_filter > 0), ValueError
    assert (fc_width > 0), ValueError
    assert 0 <= dropout_rate < 1, ValueError

    model = tf.keras.Sequential([
        tf.keras.Input(shape=image_size, name="Input"),
        *conv_relu_bn(filters=base_filter, kernel_size=(3, 3), strides=(1, 1)),
        *conv_relu_bn(filters=2*base_filter, kernel_size=(3, 3), strides=(1, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        *conv_relu_bn(filters=4 * base_filter, kernel_size=(3, 3), strides=(1, 1)),
        *conv_relu_bn(filters=4 * base_filter, kernel_size=(3, 3), strides=(1, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        * conv_relu_bn(filters=4 * base_filter, kernel_size=(3, 3), strides=(1, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        BatchNormalization(),
        Dense(fc_width, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        Dense(n_class, activation="softmax")
    ])

    if optimizer.lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learn_rate)

        model.compile(loss="categorical_crossentropy",
                      optimizer=opt, metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])

    return model