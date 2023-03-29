
#imports

import numpy as np
import Sparse_Subpixel_Convolution as SpConv

import time

import keras

class iSCAT_DataGenerator(keras.utils.Sequence):

    
    def __init__(self, batch_size=128, epoch_size=4096, res=32, frames=32, thread_count=10,
                PSF_path="../PSF_subpx_fl32.npy", exD=5000, devD=4000, exPT_cnt=1000, devPT_cnt=999, exIntensity=1.0, devIntensity=0.9, target_step=31,
                target_mode="count_particles"):
        
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.nm_per_px = 46

        self.dt = 0.1
        self.t_max = frames * self.dt

        self.psf_peak_px_sz = 32

        self.cam_fov_px = res

        self.step_cnt = int(self.t_max / self.dt)

        PSF = np.load(PSF_path)

        self.conv_calc = SpConv.ConvolutionCalculator_fl32(PSF, 1, verbose=1)
        self.thread_count = thread_count

        self.sample_sz_px = PSF.shape[2] - self.cam_fov_px
        self.FOV_edge = self.sample_sz_px / 2 - self.cam_fov_px / 2

        self.exD          = exD         
        self.devD         = devD        
        self.exPT_cnt     = exPT_cnt    
        self.devPT_cnt    = devPT_cnt   
        self.exIntensity  = exIntensity 
        self.devIntensity = devIntensity
        self.target_step  = target_step 

        self.target_mode  = target_mode
        
        print("Sample width: ", self.sample_sz_px)

        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.epoch_size / self.batch_size))

    def pt_cnt_scale(self, cnts):
        area_ratio = self.cam_fov_px ** 2 / self.sample_sz_px ** 2
        pt_cnt_mean = area_ratio * self.exPT_cnt
        aprox_pt_cnt_std = area_ratio * self.devPT_cnt

        return (cnts - pt_cnt_mean) / aprox_pt_cnt_std
        
    def pt_cnt_rescale(self, cnts):
        area_ratio = self.cam_fov_px ** 2 / self.sample_sz_px ** 2
        pt_cnt_mean = area_ratio * self.exPT_cnt
        aprox_pt_cnt_std = area_ratio * self.devPT_cnt

        return cnts * aprox_pt_cnt_std + pt_cnt_mean

    def __getitem__(self, index):
        'Generate one batch of data'
        samples, particle_positions, diffusion_coefs, pt_cnts, particles_in_sight_cnt, avg_diffusions = self.GenSamples()

        target = None
        if self.target_mode == "count_particles":
            target = particles_in_sight_cnt
            target = self.pt_cnt_scale(target)

        return samples, target

    def on_epoch_end(self):
        print("Epoch ended")


    
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

        return poss #shape: (steps, particle_count, coordinates)
    
    
    def GetTargets(self, particle_positions, diffusion_coefs, pt_cnts, step):
        xc_in_sight = (particle_positions[step, :, 0] > 0) & (particle_positions[step, :, 0] < self.cam_fov_px)
        yc_in_sight = (particle_positions[step, :, 1] > 0) & (particle_positions[step, :, 1] < self.cam_fov_px)
        pt_in_sight = xc_in_sight & yc_in_sight
        
        sample_inds = np.insert(np.cumsum(pt_cnts), 0, 0)

        particles_in_sight_cnt = list()
        avg_diffusions = list()
        for i in range(pt_cnts.shape[0]):
            particles_in_sight_cnt.append(np.sum(pt_in_sight[sample_inds[i]:sample_inds[i+1]]))
            avg_diffusions.append(np.mean(diffusion_coefs[sample_inds[i]:sample_inds[i+1]]))
            
        return particles_in_sight_cnt, avg_diffusions
    
    def GenSamples(self):
        
        exD          = self.exD         
        devD         = self.devD        
        exPT_cnt     = self.exPT_cnt    
        devPT_cnt    = self.devPT_cnt   
        exIntensity  = self.exIntensity 
        devIntensity = self.devIntensity
        target_step  = self.target_step 

        sample_cnt   = self.batch_size


        pt_cnts = np.random.uniform(exPT_cnt - devPT_cnt, exPT_cnt + devPT_cnt, (sample_cnt)).astype(np.int32)
        
        pt_cnt = np.sum(pt_cnts)
        diffusion_coefs = np.random.uniform(exD - devD, exD + devD, (pt_cnt))
        intensities = np.random.uniform(exIntensity - devIntensity, exIntensity + devIntensity, (pt_cnt)).astype(np.float32)
        
        particle_positions = self.GenParticlePositions(diffusion_coefs, pt_cnt, True)
        
        samples = self.conv_calc.convolve(self.thread_count, self.cam_fov_px, particle_positions, pt_cnts, intensities, verbose=1)
        
        #Normalization
        sample_mins = np.amin(samples, axis=(1, 2, 3))
        sample_maxs = np.amax(samples, axis=(1, 2, 3))
        samples = (samples - sample_mins[:, None, None, None]) / (sample_maxs - sample_mins)[:, None, None, None]
        
        #Generate targets
        particles_in_sight_cnt, avg_diffusions = self.GetTargets(particle_positions, diffusion_coefs, pt_cnts, target_step)
        
        return np.array(samples), particle_positions, diffusion_coefs, pt_cnts, np.array(particles_in_sight_cnt), np.array(avg_diffusions)
    