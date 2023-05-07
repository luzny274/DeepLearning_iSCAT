import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import DL_Sequence

import multiprocessing


def getDatasetGen0(epoch_size, batch_size, verbose, regen):
    num_classes = 8

    exD=5000
    devD=2000
    exPT_cnt=170
    devPT_cnt=140
    exIntensity=1.0
    devIntensity=0.4

    res=64
    frames=32
    target_frame=15
    
    number_of_threads = multiprocessing.cpu_count()

    data_generator = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=epoch_size, res=res, frames=frames, thread_count=number_of_threads * 2 // 3,
                        PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
                        num_classes=num_classes, verbose = verbose, noise_func = None, mode = "dynamic", regen = regen)

    return num_classes, frames, res, data_generator

    
def getDatasetGen1(epoch_size, batch_size, verbose, regen):
    #High resolution and lower particle density
    num_classes = 5

    exD=5000
    devD=4000
    exPT_cnt=100
    devPT_cnt=99
    exIntensity=1.0
    devIntensity=0.3

    target_frame=31
    res=64
    frames=32
    
    number_of_threads = multiprocessing.cpu_count()

    data_generator = DL_Sequence.iSCAT_DataGenerator(batch_size=batch_size, epoch_size=epoch_size, res=res, frames=frames, thread_count=int(number_of_threads * 2 / 3),
                        PSF_path="../PSF_subpx_fl32.npy", exD=exD, devD=devD, exPT_cnt=exPT_cnt, devPT_cnt=devPT_cnt, exIntensity=exIntensity, devIntensity=devIntensity, target_frame=target_frame,
                        num_classes=num_classes, verbose = verbose, noise_func = None, mode = "dynamic", regen = regen)

    return num_classes, frames, res, data_generator

def getDatasetGen(dataset, epoch_size, batch_size, verbose, regen):
    if dataset == 0:
        return getDatasetGen0(epoch_size, batch_size, verbose, regen) 
    elif dataset == 1:
        return getDatasetGen1(epoch_size, batch_size, verbose, regen) 
    else:
        print("Error: Wrong dataset number")

