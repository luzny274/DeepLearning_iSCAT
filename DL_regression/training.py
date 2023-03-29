
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import DL_Sequence

import glob
import time
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle

from ipywidgets import Layout, interact, IntSlider, FloatSlider

import tensorflow as tf

def main():
    num_classes = 16

    train_gen = DL_Sequence.iSCAT_DataGenerator(batch_size=16, epoch_size=8192, res=32, frames=32, thread_count=10,
                    PSF_path="../PSF_subpx_fl32.npy", exD=5000, devD=4000, exPT_cnt=10, devPT_cnt=9, exIntensity=1.0, devIntensity=0.9, target_frame=15,
                    num_classes=num_classes)
    # train_gen = DL_Dataset.iSCAT_DataGenerator(batch_size=1024, epoch_size=65536, res=32, frames=32, thread_count=18,
    #                 PSF_path="../PSF_subpx_fl32.npy", exD=5000, devD=4000, exPT_cnt=1000, devPT_cnt=999, exIntensity=1.0, devIntensity=0.9, target_frame=31,
    #                 target_mode="count_particles")

    test_gen = DL_Sequence.iSCAT_DataGenerator(batch_size=256, epoch_size=256, res=32, frames=32, thread_count=10,
                    PSF_path="../PSF_subpx_fl32.npy", exD=5000, devD=4000, exPT_cnt=10, devPT_cnt=9, exIntensity=1.0, devIntensity=0.9, target_frame=31,
                    num_classes=num_classes)


    seed = 42
    tf.keras.utils.set_random_seed(seed)

    hidden_layer = 500
    epochs = 4


    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[train_gen.step_cnt, train_gen.cam_fov_px, train_gen.cam_fov_px]),
        tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    optimizer = tf.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        # loss=tf.keras.losses.MeanSquaredError(),
        # loss = tf.keras.losses.Huber(),
        # metrics=[tf.keras.metrics.MeanAbsoluteError(name="accuracy")],
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    model.summary()

    model.fit(
                x=train_gen, validation_data=test_gen,
                epochs=epochs,
            )
    del train_gen
    del test_gen
if __name__ == '__main__':
    main()