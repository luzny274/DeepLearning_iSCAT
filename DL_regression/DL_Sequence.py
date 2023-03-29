
#imports

import numpy as np
import Sparse_Subpixel_Convolution as SpConv

import time

import keras
# from multiprocessing import Process, Queue
import multiprocessing

class SampleGenerator:
    def __init__(self, epoch_size, res, frames, thread_count,
                PSF_path, exD, devD, exPT_cnt, devPT_cnt, exIntensity, devIntensity, target_frame,
                num_classes, verbose):
        self.verbose = verbose
        self.epoch_size = epoch_size

        self.nm_per_px = 46

        self.dt = 0.1
        self.t_max = frames * self.dt

        self.psf_peak_px_sz = 32

        self.cam_fov_px = res

        self.step_cnt = int(self.t_max / self.dt)

        PSF = np.load(PSF_path)

        self.conv_calc = SpConv.ConvolutionCalculator_fl32(PSF, 1, verbose=self.verbose)
        self.thread_count = thread_count

        self.sample_sz_px = PSF.shape[2] - self.cam_fov_px
        self.FOV_edge = self.sample_sz_px / 2 - self.cam_fov_px / 2

        self.exD          = exD         
        self.devD         = devD        
        self.exPT_cnt     = exPT_cnt    
        self.devPT_cnt    = devPT_cnt   
        self.exIntensity  = exIntensity 
        self.devIntensity = devIntensity
        self.target_frame  = target_frame 

        self.num_classes  = num_classes
        
        print("Sample width: ", self.sample_sz_px)
        

    def GenParticlePositions(self, D : np.ndarray, particle_count : int, loop : bool):
        variance = 4 * D * self.dt / (self.nm_per_px ** 2)

        start_poss = np.random.uniform(0, self.sample_sz_px, (particle_count, 2))
        dposs = np.random.normal(0, variance[None, :, None], (self.step_cnt, particle_count, 2))

        poss = start_poss[None, :, :] + np.cumsum(dposs, axis = 0)

        s = self.sample_sz_px

        if(loop):
            poss[poss < 0] = poss[poss < 0] + np.floor(-poss[poss < 0] / s + 1) * s
            poss[poss > s] = poss[poss > s] - np.floor( poss[poss > s] / s + 0) * s

        poss -= self.FOV_edge

        return poss
    
    
    def GetTargets(self, particle_positions, pt_cnts, step):
        start = time.time()
        xc_in_sight = (particle_positions[step, :, 0] > 0) & (particle_positions[step, :, 0] < self.cam_fov_px)
        yc_in_sight = (particle_positions[step, :, 1] > 0) & (particle_positions[step, :, 1] < self.cam_fov_px)
        pt_in_sight = xc_in_sight & yc_in_sight
        
        sample_inds = np.insert(np.cumsum(pt_cnts), 0, 0)

        particles_in_sight_cnt = list()
        for i in range(pt_cnts.shape[0]):
            particles_in_sight_cnt.append(np.sum(pt_in_sight[sample_inds[i]:sample_inds[i+1]]))

        if(self.verbose > 0):
            print("Target generation time: {:.3f}s".format(time.time() - start))
            
        return particles_in_sight_cnt
    
    def GenSamples(self):
        start = time.time()
        
        exD          = self.exD         
        devD         = self.devD        
        exPT_cnt     = self.exPT_cnt    
        devPT_cnt    = self.devPT_cnt   
        exIntensity  = self.exIntensity 
        devIntensity = self.devIntensity
        target_frame = self.target_frame 

        sample_cnt   = self.epoch_size


        pt_cnts = np.random.uniform(exPT_cnt - devPT_cnt, exPT_cnt + devPT_cnt, (sample_cnt)).astype(np.int32)
        
        pt_cnt = np.sum(pt_cnts)
        diffusion_coefs = np.random.uniform(exD - devD, exD + devD, (pt_cnt))
        intensities = np.random.uniform(exIntensity - devIntensity, exIntensity + devIntensity, (pt_cnt)).astype(np.float32)
        
        particle_positions = self.GenParticlePositions(diffusion_coefs, pt_cnt, True)
        
        if self.verbose > 0:
            print("Particle generation time: {:.3f}s".format(time.time() - start))
            
        start_conv = time.time()
        samples = self.conv_calc.convolve(self.thread_count, self.cam_fov_px, particle_positions, pt_cnts, intensities, verbose=self.verbose)
        
        if self.verbose > 0:
            print("Conv generation time: {:.3f}s".format(time.time() - start_conv))
        #Normalization
        sample_mins = np.amin(samples, axis=(1, 2, 3))
        sample_maxs = np.amax(samples, axis=(1, 2, 3))
        samples = (samples - sample_mins[:, None, None, None]) / (sample_maxs - sample_mins)[:, None, None, None]
        
        #Generate targets
        particles_in_sight_cnt = self.GetTargets(particle_positions, pt_cnts, target_frame)
        
        if self.verbose > 0:
            print("Whole generation time: {:.3f}s".format(time.time() - start))

        return [np.array(samples),
                particle_positions,
                diffusion_coefs,
                pt_cnts,
                np.minimum(np.array(particles_in_sight_cnt), self.num_classes - 1)]

def sampleWorker(input_queue, output_queue, epoch_size, res, frames, thread_count,
                PSF_path, exD, devD, exPT_cnt, devPT_cnt, exIntensity, devIntensity, target_frame,
                num_classes, verbose):
    
    sampleGenerator = SampleGenerator(epoch_size, res, frames, thread_count,
                PSF_path, exD, devD, exPT_cnt, devPT_cnt, exIntensity, devIntensity, target_frame,
                num_classes, verbose)
    while True:
        input = input_queue.get()
        print(input)
        if input == "Die":
            break
        print("second", input)
        print("tady1")
        output_queue.put(sampleGenerator.GenSamples())
        print("tady2")


class iSCAT_DataGenerator(keras.utils.Sequence):
    parallel = True
    p = None
    process_running = False
    verbose = 1

    indices = None

    samples                 = None
    particle_positions      = None
    diffusion_coefs         = None
    pt_cnts                 = None
    particles_in_sight_cnt  = None

    
    def __init__(self, batch_size=128, epoch_size=4096, res=32, frames=32, thread_count=10,
                PSF_path="../PSF_subpx_fl32.npy", exD=5000, devD=4000, exPT_cnt=1000, devPT_cnt=999, exIntensity=1.0, devIntensity=0.9, target_frame=15,
                num_classes = 16):
        
        self.indices = np.array(range(epoch_size))
        
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.nm_per_px = 46
        self.dt = 0.1
        self.t_max = frames * self.dt
        self.psf_peak_px_sz = 32
        self.cam_fov_px = res
        self.step_cnt = int(self.t_max / self.dt)

        self.thread_count = thread_count

        self.exD          = exD         
        self.devD         = devD        
        self.exPT_cnt     = exPT_cnt    
        self.devPT_cnt    = devPT_cnt   
        self.exIntensity  = exIntensity 
        self.devIntensity = devIntensity
        self.target_frame = target_frame 

        self.num_classes  = num_classes
        
        if self.parallel:
            self.input_queue = multiprocessing.Queue()
            self.output_queue = multiprocessing.Queue()
            self.p = multiprocessing.Process(target=sampleWorker, args=(self.input_queue, self.output_queue, epoch_size, res, frames, thread_count,
                PSF_path, exD, devD, exPT_cnt, devPT_cnt, exIntensity, devIntensity, target_frame,
                num_classes, self.verbose))
            self.p.start()

            self.input_queue.put("Run")
            self.samples, self.particle_positions, self.diffusion_coefs, self.pt_cnts, self.particles_in_sight_cnt = self.output_queue.get()
            self.input_queue.put("Run")

            print("Min particle in sight cnt:", self.particles_in_sight_cnt.min())
            print("Max particle in sight cnt:", self.particles_in_sight_cnt.max())
            print("Avg particle in sight cnt:", self.particles_in_sight_cnt.mean())
            print("Std particle in sight cnt:", self.particles_in_sight_cnt.std())
        else:
            self.sampleGenerator = SampleGenerator(epoch_size, res, frames, thread_count,
                PSF_path, exD, devD, exPT_cnt, devPT_cnt, exIntensity, devIntensity, target_frame,
                num_classes, self.verbose)
            
            self.samples, self.particle_positions, self.diffusion_coefs, self.pt_cnts, self.particles_in_sight_cnt = self.sampleGenerator.GenSamples() 
            
            print("Min particle in sight cnt:", self.particles_in_sight_cnt.min())
            print("Max particle in sight cnt:", self.particles_in_sight_cnt.max())
            print("Avg particle in sight cnt:", self.particles_in_sight_cnt.mean())
            print("Std particle in sight cnt:", self.particles_in_sight_cnt.std())

        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.epoch_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # if self.parallel:
        #     start = time.time()
        #     if not self.p.is_alive():
        #         end = time.time()
        #         print("Check took seconds: ", end-start)
        #         print("New dataset generated! Switching...")
        #         self.SwitchDatasets()
        #         self.p = multiprocessing.Process(target=self.GenSamples)
        #         self.p.start()
        #     end = time.time()
        #     print("Took seconds: ", end-start)
        if self.parallel and self.output_queue.qsize() > 0:
            print("New dataset generated!")
            self.samples, self.particle_positions, self.diffusion_coefs, self.pt_cnts, self.particles_in_sight_cnt = self.output_queue.get()
            self.input_queue.put("Run")
        batch_begin = index * self.batch_size
        batch_end = np.minimum(self.epoch_size, (index + 1) * self.batch_size)

        batch_indices = self.indices[batch_begin:batch_end]
        return self.samples[batch_indices], self.particles_in_sight_cnt[batch_indices]

    def on_epoch_end(self):
        # print("Epoch ended")
        np.random.shuffle(self.indices)

    def __del__(self):
        # print("here1")
        # while self.output_queue.qsize() < 1:
        #     print("Waiting...")
        # self.samples, self.particle_positions, self.diffusion_coefs, self.pt_cnts, self.particles_in_sight_cnt = self.output_queue.get()
        # print("here2")
        self.input_queue.put("Die")
        # print("here3")

    