
import DL_SampleGeneration

import glob
import time
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle

from ipywidgets import Layout, interact, IntSlider, FloatSlider

gen = DL_SampleGeneration.SampleGenerator()

exD = 5000
devD = 2500

exPT_cnt = 100
devPT_cnt = 99

sample_cnt = 100000

exIntensity = 1.0
devIntensity = 0.7

step_cnt = gen.step_cnt
res = gen.cam_fov_px

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
import tensorflow as tf

seed = 42
tf.keras.utils.set_random_seed(seed)

hidden_layer = 1000
batch_size = 32
epochs = 10


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[step_cnt, res, res]),
    tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
])

optimizer = tf.optimizers.Adam()

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MeanSquaredError(),
)

timing = time.time()
samples, particle_positions, diffusion_coefs, pt_cnts, particles_in_sight_cnt, avg_diffusions = gen.GenSamples(exD, devD, exPT_cnt, devPT_cnt, sample_cnt, exIntensity, devIntensity, gen.step_cnt-1)
print("Sample generation took: {:.3f}s".format(time.time() - timing))

samples = np.array(samples)
particles_in_sight_cnt = np.array(particles_in_sight_cnt)
avg_diffusions = np.array(avg_diffusions)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

train_gen = DataGenerator(samples, avg_diffusions, batch_size)

model.fit(
    train_gen,
    epochs=epochs
)