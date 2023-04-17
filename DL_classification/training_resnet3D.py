
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import iSCAT_Datasets

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

import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=0, type=int, help="Dataset (0/1/2)")
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


class ResNet3D(Model):
    def _cnn(self, inputs, filters, kernel_size, stride, activation):
        hidden = tf.keras.layers.Conv3D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden) if activation else hidden
        return hidden

    def _block(self, inputs, filters, stride, layer_index):
        hidden = self._cnn(inputs, filters, self.kernel_size, stride, activation=True)
        hidden = self._cnn(hidden, filters, self.kernel_size, 1, activation=False)
        if stride > 1:
            residual = self._cnn(inputs, filters, 1, stride, activation=False)
        else:
            residual = inputs
        hidden = residual + hidden
        hidden = self._activation(hidden)
        return hidden

    def __init__(self, shape, num_classes, activation, depth, filters_start, kernel_size):
        self.shape = shape
        self.num_classes = num_classes
        self.activation = activation
        self.depth = depth

        self.filters_start = filters_start
        self.kernel_size = kernel_size

        n = (depth - 2) // 6


        inputs = tf.keras.Input(shape=shape, dtype=tf.float32)
        hidden = tf.keras.layers.Reshape((1, shape[0], shape[1], shape[2]), input_shape=shape)(inputs)
        hidden = self._cnn(hidden, filters_start, kernel_size, 1, activation=True)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, filters_start * (1 << stage), 2 if stage > 0 and block == 0 else 1, (stage * n + block) / (3 * n - 1))
        hidden = tf.keras.layers.GlobalAvgPool3D()(hidden)
        outputs = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)

def main(args):
    epochs = 100
    batch_size = 16
    train_epoch_size = 16384
    test_epoch_size = 8192

    seed = int(datetime.datetime.now().timestamp()) % 1000000
    tf.keras.utils.set_random_seed(seed)

    tf.keras.backend.image_data_format()

    print(tf.keras.backend.image_data_format())
    tf.keras.backend.set_image_data_format('channels_first')
    print(tf.keras.backend.image_data_format())

    mode = "dynamic"
    print("Mode: " + mode)

    num_classes, frames, res, test_gen = iSCAT_Datasets.getDatasetGen(args.dataset, test_epoch_size, batch_size, verbose=0, mode=mode, regen=False)

    activation = "swish"
    depth = 62
    filters_start = 16
    kernel_size = (5, 3, 3)

    model_comment = "_" + mode
    model_name = "models/model_resnet3d_tk" + str(kernel_size[0]) + model_comment

    if args.finetune or args.evaluate:
        model = tf.keras.models.load_model(model_name + ".h5", custom_objects={"ResNet3D" : ResNet3D})
    else:
        model = ResNet3D((frames, res, res), num_classes, activation, depth, filters_start, kernel_size)


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

    with open('resnet3dsummary.txt','w') as f:
        print("", file=f)

    def myprint(s):
        with open('resnet3dsummary.txt','a') as f:
            print(s, file=f)

    model.summary(print_fn=myprint)

    # test_gen = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=test_epoch_size, res=res, frames=frames, thread_count=20,
    #                 PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
    #                 num_classes=num_classes, verbose = 0, noise_func = None, mode = mode, regen = False)

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

        # train_gen = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=train_epoch_size, res=res, frames=frames, thread_count=20,
        #                 PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
        #                 num_classes=num_classes, verbose = 1, noise_func = None, mode = mode, regen = True)
        num_classes, frames, res, train_gen = iSCAT_Datasets.getDatasetGen(args.dataset, train_epoch_size, batch_size, verbose=1, mode=mode, regen=True)
                        

        save_best_callback = SaveBestModel(name=model_name, metric="val_loss", this_max=False)
        model.fit(x=train_gen, validation_data=test_gen, epochs=epochs, callbacks=[save_best_callback])
                
        train_gen.destroy()

    test_gen.destroy()

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)