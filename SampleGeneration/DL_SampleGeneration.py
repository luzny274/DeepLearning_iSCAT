
#imports

import numpy as np
import Sparse_Subpixel_Convolution as SpConv

import time
import gc

class SampleGenerator:
    def __init__(self):
        self.nm_per_px = 46

        self.dt = 0.1
        self.t_max = 3.2

        self.psf_peak_px_sz = 32

        self.cam_fov_px = 32

        self.step_cnt = int(self.t_max / self.dt)


        self.PSF = np.load("../PSF_subpx_fl32.npy")

        self.sample_sz_px = self.PSF.shape[2] - self.cam_fov_px
        self.FOV_edge = self.sample_sz_px / 2 - self.cam_fov_px / 2
        
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
    
    def GenSamples(self, exD, devD, exPT_cnt, devPT_cnt, sample_cnt, exIntensity, devIntensity, target_step):
        pt_cnts = np.random.uniform(exPT_cnt - devPT_cnt, exPT_cnt + devPT_cnt, (sample_cnt)).astype(np.int32)
        
        pt_cnt = np.sum(pt_cnts)
        diffusion_coefs = np.random.uniform(exD - devD, exD + devD, (pt_cnt))
        intensities = np.random.uniform(exIntensity - devIntensity, exIntensity + devIntensity, (pt_cnt)).astype(np.float32)
        
        particle_positions = self.GenParticlePositions(diffusion_coefs, pt_cnt, True)
        
        samples = SpConv.convolve(10, self.cam_fov_px, particle_positions, pt_cnts, intensities, self.PSF, np.float32, verbose=0)
        
        #Normalization
        sample_mins = np.amin(samples, axis=(1, 2, 3))
        sample_maxs = np.amax(samples, axis=(1, 2, 3))
        samples = (samples - sample_mins[:, None, None, None]) / (sample_maxs - sample_mins)[:, None, None, None]
        
        #Generate targets
        particles_in_sight_cnt, avg_diffusions = self.GetTargets(particle_positions, diffusion_coefs, pt_cnts, target_step)
        
        return samples, particle_positions, diffusion_coefs, pt_cnts, particles_in_sight_cnt, avg_diffusions
    


        
        
        
        
        