
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
        # self.model.save_weights(self.name, include_optimizer=False)


    def on_epoch_end(self, epoch, logs=None):
        if self.max * logs[self.metric] > self.max * self.get_best()[self.metric]:
            print("\tSaving...")
            self.set_best(logs)


def main(args):

    seed = int(datetime.datetime.now().timestamp()) % 1000000
    tf.keras.utils.set_random_seed(seed)

    hidden_layer = 5000

    print(tf.keras.backend.image_data_format())
    tf.keras.backend.set_image_data_format('channels_first')
    print(tf.keras.backend.image_data_format())
    
    model_name = "models/model_mlp" + "_dataset-" + str(args.dataset)

    if args.finetune or args.evaluate:
        model = tf.keras.models.load_model(model_name + ".h5")
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=[frames, res, res]),
            tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
        ])

    decay_steps = train_epoch_size / batch_size * epochs
    start_lr = 0.00005
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

    model.summary()
    
    number_of_threads = multiprocessing.cpu_count()

    num_classes, frames, res, test_gen = iSCAT_Datasets.getDatasetGen(args.dataset, test_epoch_size, batch_size, verbose=0, regen=False)

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

        train_gen = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=train_epoch_size, res=res, frames=frames, thread_count = int(number_of_threads * 2 / 3),
                        PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
                        num_classes=num_classes, verbose = 1, noise_func = None, mode = mode, regen = True)
                        

        save_best_callback = SaveBestModel(name=model_name, metric="val_loss", this_max=False)
        model.fit(x=train_gen, validation_data=test_gen, epochs=epochs, callbacks=[save_best_callback])
                
        train_gen.destroy()

    test_gen.destroy()

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)