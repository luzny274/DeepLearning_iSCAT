
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


class ResNet2p1D(Model):
    def _cnn_1(self, inputs, filters, stride, activation):
        hidden = inputs
        hidden = tf.keras.layers.Conv3D(filters, kernel_size=1, strides=stride, padding="same", use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden) if activation else hidden
        return hidden

    def _cnn(self, inputs, filters, stride, activation):
        hidden = inputs
        hidden = tf.keras.layers.Conv3D(filters, kernel_size=self.spatial_kernel_size, strides=(1, stride, stride), padding="same", use_bias=False)(hidden)
        hidden = tf.keras.layers.Conv3D(filters, kernel_size=self.temporal_kernel_size, strides=(stride, 1, 1), padding="same", use_bias=False)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden) if activation else hidden
        return hidden

    def _block(self, inputs, filters, stride, layer_index):
        hidden = self._cnn(inputs, filters, stride, activation=True)
        hidden = self._cnn(hidden, filters, 1, activation=False)

        # print("hidden", hidden)
        
        if stride > 1:
            residual = self._cnn_1(inputs, filters, stride, activation=False)
        else:
            residual = inputs

        # print("residual", residual)

        hidden = residual + hidden
        hidden = self._activation(hidden)
        return hidden

    def __init__(self, shape, num_classes, activation, depth, filters_start, temporal_ker, spatial_ker):
        self.shape = shape
        self.num_classes = num_classes
        self.activation = activation
        self.depth = depth

        self.filters_start = filters_start
        self.temporal_ker = temporal_ker
        self.spatial_ker = spatial_ker

        n = (depth - 2) // 6

        self.temporal_kernel_size = (temporal_ker, 1, 1)
        self.spatial_kernel_size = (1, spatial_ker, spatial_ker)

        whole_kernel = (temporal_ker, spatial_ker, spatial_ker)


        inputs = tf.keras.Input(shape=shape, dtype=tf.float32)
        hidden = tf.keras.layers.Reshape((1, shape[0], shape[1], shape[2]), input_shape=shape)(inputs)
        hidden = self._cnn_1(hidden, filters_start, 1, activation=True)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, filters_start * (1 << stage), 2 if stage > 0 and block == 0 else 1, (stage * n + block) / (3 * n - 1))
        hidden = tf.keras.layers.GlobalAvgPool3D()(hidden)
        outputs = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)

def getDatasetGen0(epoch_size, batch_size, verbose, mode, regen):
    #Low resolution and high particle density
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
    
    number_of_threads = multiprocessing.cpu_count()

    data_generator = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=epoch_size, res=res, frames=frames, thread_count=int(number_of_threads * 2 / 3),
                        PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
                        num_classes=num_classes, verbose = verbose, noise_func = None, mode = mode, regen = regen)

    return num_classes, frames, res, data_generator

    
def getDatasetGen1(epoch_size, batch_size, verbose, mode, regen):
    #High resolution and low particle density
    num_classes = 5

    exD=5000
    devD=4000
    exPT_cnt=100
    devPT_cnt=99
    exIntensity=1.0
    devIntensity=0.3

    target_frame=31
    res=64
    frames=64
    
    number_of_threads = multiprocessing.cpu_count()

    data_generator = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=epoch_size, res=res, frames=frames, thread_count=int(number_of_threads * 2 / 3),
                        PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
                        num_classes=num_classes, verbose = verbose, noise_func = None, mode = mode, regen = regen)

    return num_classes, frames, res, data_generator

def getDatasetGen(args, epoch_size, batch_size, verbose, mode, regen):
    if args.dataset == 0:
        return getDatasetGen0(epoch_size, batch_size, verbose, mode, regen) 
    elif args.dataset == 1:
        return getDatasetGen1(epoch_size, batch_size, verbose, mode, regen) 
    else:
        print("Error: Wrong dataset number")

def main(args):
    epochs = 200
    batch_size = 32
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

    num_classes, frames, res, test_gen = getDatasetGen(args, test_epoch_size, batch_size, verbose=0, mode=mode, regen=False)

    activation = "swish"
    depth = 52
    filters_start = 16

    spatial_kernel_size = 3
    temporal_kernel_size = 3

    model_comment = "_dataset-" + str(args.dataset)
    model_name = "models/model_resnet(2+1)d_tk" + str(temporal_kernel_size) + model_comment

    if args.finetune or args.evaluate:
        model = tf.keras.models.load_model(model_name + ".h5")
    else:
        model = ResNet2p1D((frames, res, res), num_classes, activation, depth, filters_start, temporal_kernel_size, spatial_kernel_size)


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

    with open('resnet(2+1)dsummary.txt','w') as f:
        print("", file=f)

    def myprint(s):
        with open('resnet(2+1)dsummary.txt','a') as f:
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
        num_classes, frames, res, train_gen = getDatasetGen(args, train_epoch_size, batch_size, verbose=1, mode=mode, regen=True)
                        

        save_best_callback = SaveBestModel(name=model_name, metric="val_loss", this_max=False)
        model.fit(x=train_gen, validation_data=test_gen, epochs=epochs, callbacks=[save_best_callback])
                
        train_gen.destroy()

    test_gen.destroy()

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)