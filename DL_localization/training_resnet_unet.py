
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

        # self.model.save(model_filename, include_optimizer=False)
        self.model.save_weights(self.name + "_weights/")


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


class ResNet(Model):
    def _cnn(self, inputs, filters, kernel_size, stride, activation):
        hidden = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
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
        hidden = self._cnn(inputs, filters_start, kernel_size, 1, activation=True)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, filters_start * (1 << stage), 2 if stage > 0 and block == 0 else 1, (stage * n + block) / (3 * n - 1))
        hidden = tf.keras.layers.GlobalAvgPool2D()(hidden)
        outputs = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)


def main(args):
    epochs = 30
    batch_size = 32
    train_epoch_size = 16384
    test_epoch_size = 8192
    # train_epoch_size = 256
    # test_epoch_size = 128

    seed = int(datetime.datetime.now().timestamp()) % 1000000
    tf.keras.utils.set_random_seed(seed)

    tf.keras.backend.image_data_format()

    print(tf.keras.backend.image_data_format())
    tf.keras.backend.set_image_data_format('channels_first')
    print(tf.keras.backend.image_data_format())

    mode = "dynamic"
    print("Mode: " + mode)
    print("Dataset: " + str(args.dataset))

    num_classes, frames, res, test_gen = iSCAT_Datasets.getDatasetGen(args.dataset, test_epoch_size, batch_size, verbose=0, mode=mode, regen=False)


    activation = "swish"
    depth = 62
    kernel_size = 3

    backbone_comment = "_dataset-" + str(args.dataset)
    backbone_name = "backbones/model_resnet_k" + str(kernel_size) + backbone_comment

    backbone = ResNet((frames, res, res), num_classes, activation, depth, frames, kernel_size)
    backbone.load_weights(backbone_name + "_weights/")
    backbone.trainable = False

    # (128, 16, 16), (64, 32, 32), (32, 64, 64)
    backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
                "activation_60", "activation_40", "activation_20"]]
    )

    filters = [128, 64, 32]
    hidden = backbone.outputs[0]
    cnn_blocks = 30
    for ii in range(len(backbone.outputs) - 1):
        conv = backbone.outputs[ii + 1]
        channels = filters[ii + 1]

        conv = tf.keras.layers.Conv2D(channels, kernel_size=1)(conv)
        hidden = tf.keras.layers.Conv2DTranspose(channels, kernel_size=2, strides=2, padding="same")(hidden)
        hidden = tf.keras.layers.Concatenate(axis=1)([hidden, conv])
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        hidden = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        for i in range(cnn_blocks):
            residual = hidden
            hidden = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            hidden = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden += residual
            hidden = tf.keras.layers.ReLU()(hidden)

    outputs = tf.keras.layers.Conv2D(1, kernel_size=3, padding="same", activation=tf.nn.sigmoid)(hidden)

    model = tf.keras.Model(inputs=backbone.inputs, outputs=outputs)


    loss_type = "bce"
    label_smoothing = 0.0

    if loss_type == "bce":
        loss=tf.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    elif loss_type == "iou":
        def loss(y_t, y_p):
            return tf.math.reduce_mean(1 - ((tf.math.reduce_sum(y_t * y_p, axis=[1,2]) + 1) /
                                            (tf.math.reduce_sum(y_t + y_p - y_t * y_p, axis=[1,2]) + 1)))
    elif loss_type == "dice":
        def loss(y_t, y_p):
            return tf.math.reduce_mean(1 - (2 * (tf.math.reduce_sum(y_t * y_p, axis=[1,2]) + 1) /
                                            (tf.math.reduce_sum(y_t + y_p, axis=[1,2]) + 1)))
    else:
        raise ValueError("Unsupported loss '{}'".format(loss_type))

    frozen_epochs = int(epochs / 2)
    finetune_epochs = int(epochs / 2)
    
    decay_steps = int(train_epoch_size / batch_size * frozen_epochs)

    num_classes, frames, res, train_gen = iSCAT_Datasets.getDatasetGen(args.dataset, train_epoch_size, batch_size, verbose=1, mode=mode, regen=True)
                    
    model_name = "models/resnet_unet"

    save_best_callback = SaveBestModel(name=model_name, metric="val_loss", this_max=False)
    
    learning_rate = tf.optimizers.schedules.CosineDecay(0.0001, decay_steps, 0.0)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate, jit_compile=False),
        loss=loss,
        metrics=[tf.metrics.BinaryAccuracy(name="accuracy")],
    )
    with open('resnet_frozen_unet_summary.txt','w') as f:
        print("", file=f)

    def myprint(s):
        with open('resnet_frozen_unet_summary.txt','a') as f:
            print(s, file=f)
    model.summary(print_fn=myprint)
    model.fit(x=train_gen, validation_data=test_gen, epochs=frozen_epochs, callbacks=[save_best_callback])
          
    backbone.trainable = True
    learning_rate = tf.optimizers.schedules.CosineDecay(0.0001, decay_steps, 0.0)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate, jit_compile=False),
        loss=loss,
        metrics=[tf.metrics.BinaryAccuracy(name="accuracy")],
    )
    with open('resnet_fntn_unet_summary.txt','w') as f:
        print("", file=f)

    def myprint(s):
        with open('resnet_fntn_unet_summary.txt','a') as f:
            print(s, file=f)
    model.summary(print_fn=myprint)
    model.fit(x=train_gen, validation_data=test_gen, epochs=finetune_epochs, callbacks=[save_best_callback])
            
    train_gen.destroy()

    test_gen.destroy()

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)