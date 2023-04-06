
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import DL_Sequence

import argparse

import glob
import time
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle

import datetime
from ipywidgets import Layout, interact, IntSlider, FloatSlider

import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--finetune", default=False, action="store_true", help="Train new/finetune")
parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate/train")

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, name="model_seed", metric="val_accuracy", this_max=True):
        self.name = name
        self.metric = metric
        self.max = (float(this_max) - 0.5) * 2.0

    def get_best(self):
        filename = self.name + ".txt"
        vals_dic = {}

        if os.path.isfile(filename):
            f = open(filename, "r")
            vals = f.readline().split(" ; ")
            f.close()
            vals_dic = {"val_loss": float(vals[0]),  "val_accuracy": float(vals[1])}
        else:
            vals_dic = {"val_loss": float('inf'),  "val_accuracy": float('-inf')}

        return vals_dic
        
    def set_best(self, vals):
        filename = self.name + ".txt"
        model_filename = self.name + ".h5"

        f = open(filename, "w")
        f.write(str(vals['val_loss']) + " ; " + str(vals['val_accuracy']))
        f.close()

        self.model.save(model_filename, include_optimizer=False)


    def on_epoch_end(self, epoch, logs=None):
        if self.max * logs[self.metric] > self.max * self.get_best()[self.metric]:
            print("\tSaving...")
            self.set_best(logs)


class Model(tf.keras.Model):
    def _activation(self, inputs):
        if self.activation == "relu":
            return tf.keras.layers.Activation(tf.nn.relu)(inputs)
        if self.activation == "lrelu":
            return tf.keras.layers.Activation(tf.nn.leaky_relu)(inputs)
        if self.activation == "elu":
            return tf.keras.layers.Activation(tf.nn.elu)(inputs)
        if self.activation == "swish":
            return tf.keras.layers.Activation(tf.nn.swish)(inputs)
        if self.activation == "gelu":
            return tf.keras.layers.Activation(tf.nn.gelu)(inputs)
        raise ValueError("Unknown activation '{}'".format(self.activation))

class WideNet(Model):
    def _cnn(self, inputs, filters, kernel_size, stride):
        return tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)

    def _bn_activation(self, inputs):
        return self._activation(tf.keras.layers.BatchNormalization()(inputs))

    def _block(self, inputs, filters, stride, layer_index):
        hidden = self._bn_activation(inputs)
        residual = hidden if stride > 1 or filters != inputs.shape[-1] else inputs
        hidden = self._cnn(hidden, filters, self.kernel_size, stride)
        hidden = self._bn_activation(hidden)
        hidden = self._cnn(hidden, filters, self.kernel_size, 1)

        if stride > 1 or filters != inputs.shape[-1]:
            residual = self._cnn(residual, filters, 1, stride)
        hidden = residual + hidden
        return hidden

    def __init__(self, shape, num_classes, activation, depth, width, filters_start, kernel_size):
        self.shape = shape
        self.num_classes = num_classes
        self.activation = activation
        self.depth = depth
        self.width = width

        self.filters_start = filters_start
        self.kernel_size = kernel_size

        n = (depth - 4) // 6

        inputs = tf.keras.Input(shape=shape, dtype=tf.float32)
        hidden = self._cnn(inputs, filters_start * width, kernel_size, 1)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, filters_start * width * (1 << stage), 2 if stage > 0 and block == 0 else 1, (stage * n + block) / (3 * n - 1))
        hidden = self._bn_activation(hidden)
        hidden = tf.keras.layers.GlobalAvgPool2D()(hidden)
        outputs = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)

def main(args):
    num_classes = 5

    exD=5000
    devD=4000
    exPT_cnt=500
    devPT_cnt=499
    exIntensity=1.0
    devIntensity=0.3
    target_frame=15
    res=32
    frames=32

    tf.config.run_functions_eagerly(False)
    print("Executing eagerly: ", tf.executing_eagerly())

    train_epoch_size = 16384
    test_epoch_size = 8192
    batch_size = 32
    epochs = 200

    seed = int(datetime.datetime.now().timestamp()) % 1000000
    tf.keras.utils.set_random_seed(seed)

    print(tf.keras.backend.image_data_format())
    tf.keras.backend.set_image_data_format('channels_first')
    print(tf.keras.backend.image_data_format())

    mode = "dynamic"
    print("Mode: " + mode)

    activation = "relu"
    depth = 154
    width = 1
    kernel_size = 3

    model_comment = "_" + mode
    model_name = "models/model_widenet_k" + str(kernel_size)  + "_w" + str(width) + model_comment

    if args.finetune or args.evaluate:
        model = tf.keras.models.load_model(model_name + ".h5")
    else:
        model = WideNet((frames, res, res), num_classes, activation, depth, width, frames, kernel_size)

    decay_steps = train_epoch_size / batch_size * epochs
    start_lr = 0.0001
    end_lr = 0.0
    alpha = end_lr / start_lr
    learning_rate = tf.optimizers.schedules.CosineDecay(start_lr, decay_steps, alpha)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def mean_deviation(y_true, y_pred):
        pred_class = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.float32)
        deviations = tf.abs(y_true[:, 0] - pred_class)
        return tf.reduce_mean(deviations)

    def std_deviation(y_true, y_pred):
        pred_class = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.float32)
        deviations = tf.abs(y_true[:, 0] - pred_class)
        return tf.math.reduce_std(deviations)


    with open('widenetsummary.txt','w') as f:
        print("", file=f)
        
    def myprint(s):
        with open('widenetsummary.txt','a') as f:
            print(s, file=f)

    model.summary(print_fn=myprint)


    test_gen = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=test_epoch_size, res=res, frames=frames, thread_count=20,
                    PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
                    num_classes=num_classes, verbose = 0, noise_func = None, mode = mode, regen = False)

    if args.evaluate:
        model.compile(
            optimizer=optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy"), mean_deviation, std_deviation],
        )
        model.evaluate(test_gen)
    else:
        model.compile(
            optimizer=optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        train_gen = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=train_epoch_size, res=res, frames=frames, thread_count=20,
                        PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
                        num_classes=num_classes, verbose = 1, noise_func = None, mode = mode, regen = True)
                        

        save_best_callback = SaveBestModel(name=model_name, metric="val_loss", this_max=False)
        model.fit(x=train_gen, validation_data=test_gen, epochs=epochs, callbacks=[save_best_callback])
                
        train_gen.destroy()

    test_gen.destroy()

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)