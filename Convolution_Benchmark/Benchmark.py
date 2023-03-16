
#imports

import numpy as np
import Sparse_Subpixel_Convolution as SpConv

from PIL import Image
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle

from ipywidgets import Layout, interact, IntSlider, FloatSlider
import time
import gc


#Consts
nanometers_per_pixel = 46

dt = 0.1
t_max = 6.4

psf_peak_pixel_size = 32

camera_fov_px = 64

steps = int(t_max / dt)

exD = 5000.0
exParticle_count = 1000


#Load PSF
print("Loading PSF...")
PSF_subpx_fl32 = np.load("../PSF_subpx_fl32.npy")
sample_size_px = PSF_subpx_fl32.shape[2] - camera_fov_px
print("Sample width: ", sample_size_px)

#
FOV_edge = sample_size_px / 2 - camera_fov_px / 2

sample_count = 1000
    
def GenParticlePositions(D : float, particle_count : int, step_count : int, loop : bool):
    variance = 4 * D * dt / (nanometers_per_pixel ** 2)

    start_poss = np.random.uniform(0, sample_size_px, (particle_count, 2))
    dposs = np.random.normal(0, variance, (step_count, particle_count, 2))

    poss = start_poss[None, :, :] + np.cumsum(dposs, axis = 0)
    
    s = sample_size_px
    
    if(loop):
        poss[poss < 0] = poss[poss < 0] + np.floor(-poss[poss < 0] / s + 1) * s
        poss[poss > s] = poss[poss > s] - np.floor( poss[poss > s] / s + 0) * s
    
    poss -= FOV_edge
    
    return poss #shape: (steps, particle_count, coordinates)

sz = camera_fov_px
subpixels = PSF_subpx_fl32.shape[0]



import multiprocessing
number_of_threads = multiprocessing.cpu_count()
print(number_of_threads)

threads_begin = 1
threads_end = number_of_threads

#Prepare memory
print("Preparing memory...")
optimized_thread_cnt = 1
conv_calc = SpConv.ConvolutionCalculator_fl32(PSF_subpx_fl32, optimized_thread_cnt, 1)
###Benchmark

startTime = time.time()

particle_count = exParticle_count * sample_count

poss = GenParticlePositions(exD, particle_count, steps, True)
sample_sizes = np.full((sample_count), exParticle_count, dtype=np.int32)

print("Particle generation done in " + str(time.time() - startTime) + " s")


#Convolution

def test_convolution_parallel_omp(thread_count, particle_positions, step_count, particle_count, datatyp):    
    intensities = np.full((particle_count), 1, dtype=datatyp)

    startTime = time.time()     

    samples = conv_calc.convolve(thread_count, camera_fov_px, poss, sample_sizes, intensities, verbose=1)
    
    perf = time.time() - startTime
    return [samples, perf]


#Testing

# datatypes = [np.float64, np.float32, np.uint32, np.uint16]
# str_types = ["fl64", "fl32", "ui32", "ui16"]
datatypes = [np.float32]
str_types = ["fl32"]
perfs_omp = np.zeros((len(datatypes), threads_end - threads_begin))

sampless = list()

for d in range(len(datatypes)):
    print("Datatype: " + str_types[d])
    for i in range(threads_begin, threads_end):
        samples, perfs_omp[d, i-threads_begin] = test_convolution_parallel_omp(i, poss, steps, particle_count, datatypes[d])
        print("\t Threads: " + str(i) + "--- Convolutions done in " + str(perfs_omp[d, i-threads_begin]) + " s")
    sampless.append(samples)

    plt.plot(perfs_omp[d])
plt.legend(str_types)

del conv_calc
plt.show()