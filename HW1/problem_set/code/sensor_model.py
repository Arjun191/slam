'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # original parameters
        self._z_hit = 10
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        # modified params
        self._z_hit = 15
        self._z_short = 2
        self._z_max = 1.5 
        self._z_rand = 500

        self._map_path = '../data/map/wean.dat'
        self.map_reader = MapReader(self._map_path)
        self._map = self.map_reader.get_map()

        # original
        # self._sigma_hit = 50

        # good
        self._sigma_hit = 100
        self._lambda_short = 15

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        self._hash_map = np.load('../data/map/hash_map_v2.npy')
              
    
    def hit_gaussian(self, z_t_k, z_t_k_star):
        constant = 1 / (np.sqrt(2*np.pi*(self._sigma_hit**2)))
        exp = np.exp(-0.5*((z_t_k - z_t_k_star)**2)/(self._sigma_hit**2))
        normalizer = 1
        return normalizer*constant*exp
    
    def exp_short(self, z_t_k, z_t_k_star):
        normalizer = 1/(1 - np.exp(-1*self._lambda_short*z_t_k_star))
        exp = np.exp(-1*self._lambda_short*z_t_k)
        prob_short = normalizer*self._lambda_short*exp
        return prob_short        

    
    def beam_range_finder_model_vectorized(self, z_t1_arr, x_t1_vector):
        """
        Vectorized version of beam_range_finder_model.
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1_vector : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        
        # Initialize wt_vector as ones (shape (n, 1), where n = x_t1_vector.shape[0])
        wt_vector = np.ones((x_t1_vector.shape[0], 1))
        
        # Pre-compute x_t1_vector's indexing
        sensor_x = 25 * np.cos(np.degrees(x_t1_vector[:,2])) + x_t1_vector[:, 0]
        sensor_y = 25 * np.sin(np.degrees(x_t1_vector[:,2])) + x_t1_vector[:, 1]
        # sensor_x = x_t1_vector[:, 0]
        # sensor_y = x_t1_vector[:, 1]

        # Number of particles and measurements
        n = x_t1_vector.shape[0]  # Number of particles
        m = int(180/self._subsampling)
        sub_idx = np.arange(0, 180, self._subsampling)

        x_t1_floor_x = np.floor(sensor_x / 10).astype(int)
        x_t1_floor_y = np.floor(sensor_y / 10).astype(int)
        x_t1_deg_theta = np.degrees(x_t1_vector[:, 2])
        scan_range = np.linspace(-90, 90, 180)
        sub_scan_range = scan_range[sub_idx]
        # Ensure that for every x_t1_floor_x and x_t1_floor_y pair, there are 180 z_t_star values
        z_t_star = np.array([[self._hash_map[x, y][int((ray+theta)%360)] for ray in sub_scan_range] for x, y, theta in zip(x_t1_floor_x, x_t1_floor_y, x_t1_deg_theta)])
        z_t_star[z_t_star < 1e-6] = 1e-6 # prevents divide by zero

        # Initialize a tensor to hold (p_hit, p_short, p_max, p_rand) for each particle and measurement
        log_q_tensor  = np.zeros((n, m, 4))

        # Subsample from z_t1_arr and reshape to (1,m) for broadcasting
        z_t1_sub = z_t1_arr[sub_idx].reshape(1, m)
        z_t1_sub = np.repeat(z_t1_sub, n, axis=0)

        # p_hit condition
        log_q_tensor[:, :, 0] = np.where(
            (z_t1_sub <= self._max_range) & (z_t1_sub >= 0), 
            self.hit_gaussian(z_t1_sub, z_t_star), 
            0
        )

        # p_short condition
        log_q_tensor[:, :, 1] = np.where(
            (z_t1_sub >= 0) & (z_t1_sub <= z_t_star), 
            self.exp_short(z_t1_sub, z_t_star), 
            0
        )

        # p_max condition
        log_q_tensor[:, :, 2] = np.where(
            z_t1_sub >= self._max_range, 
            1, 
            0
        )

        # p_rand condition
        log_q_tensor[:, :, 3] = np.where(
            (z_t1_sub >= 0) & (z_t1_sub <= self._max_range), 
            1 / self._max_range, 
            0
        )

        # Weights for the 4 probabilities
        weights = np.array([self._z_hit, self._z_short, self._z_max, self._z_rand]).reshape((1,1,4))  # Shape (4,)        

        # Weighted sum across the last dimension (reduces (n, m, 4) to (n, m))
        prob_zt1 = np.sum(log_q_tensor * weights, axis=2)
        prob_zt1 = np.log(prob_zt1)

        # Take the product across the measurements (m) to reduce from (n, m) to (n, 1)
        final_prob = np.sum(prob_zt1, axis=1).reshape(n, 1)
        wt_vector = np.exp(final_prob)

        return wt_vector
    
    def beam_range_finder_model(self, z_t1_arr, x_t1, debug=False):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t

        Note: this version does not utilize subsampling
        """

        log_q = 0
        # debug = False
        if debug:
            fig, (ax1, ax2) = plt.subplots(ncols=2)

        for i, z_t in enumerate(z_t1_arr):
            deg = int(np.degrees(x_t1[2])-90+i) % 360

            # add 25 cm offset to center of range sensor
            sensor_x = 25 * np.cos(np.degrees(deg)) + x_t1[0]
            sensor_y = 25 * np.sin(np.degrees(deg)) + x_t1[1]
            z_t_star = self._hash_map[int(sensor_x/10)][int(sensor_y/10)][deg]
            # z_t_star = min(z_t_star, self._max_range)
            # z_t = min(z_t, self._max_range)
            if debug:
                ax1.plot([0, z_t * np.sin(np.radians(deg))], 
                        [0, z_t * np.cos(np.radians(deg))])

                ax2.plot([0, z_t_star * np.sin(np.radians(deg))], 
                        [0, z_t_star * np.cos(np.radians(deg))])

            if 0 <= z_t <= self._max_range:
                p_hit = self.hit_gaussian(z_t, z_t_star)
            else:
                p_hit = 0
            if 0 <= z_t <= z_t_star:
                p_short = self.exp_short(z_t, z_t_star)
            else:
                p_short = 0
            if z_t >= self._max_range:
                p_max = 1
            else:
                p_max = 0
            if 0 <= z_t <= self._max_range:
                p_rand = 1/self._max_range
            else:
                p_rand = 0
            prob_z_t = self._z_hit*p_hit + self._z_short*p_short + self._z_max*p_max + self._z_rand*p_rand
            assert prob_z_t > 0, prob_z_t
            log_q += np.log(prob_z_t)
        
        prob_zt1 = np.exp(log_q)
        
        if debug:
            ax1.set_xlabel("z_t")
            ax1.set_aspect("equal")
            ax1.grid()
            ax1.set_xlim([-self._max_range, self._max_range])
            ax1.set_ylim([-self._max_range, self._max_range])

            ax2.set_xlabel("z_t_star")
            ax2.set_aspect("equal")
            ax2.grid()
            ax2.set_xlim([-self._max_range, self._max_range])
            ax2.set_ylim([-self._max_range, self._max_range])

            plt.show(block=True)
        return prob_zt1
