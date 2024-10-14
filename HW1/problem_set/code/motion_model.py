'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # original parameters
        # self._alpha1 = 0.01
        # self._alpha2 = 0.01
        # self._alpha3 = 0.01
        # self._alpha4 = 0.01

        # modified params
        self._alpha1 = 0.0005
        self._alpha2 = 0.0005
        self._alpha3 = 0.001
        self._alpha4 = 0.001

    def sample(self, value):
        return np.random.normal(0, np.sqrt(value))
    
    def update_vectorized(self, u_t0, u_t1, x_t0_vector):
        del_rot1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        del_trans = np.sqrt(((u_t0[0] - u_t1[0])**2) + ((u_t0[1] - u_t1[1])**2))
        del_rot2 = u_t1[2] - u_t0[2] - del_rot1
        
        n = x_t0_vector.shape[0]
        del_rot1_hat = del_rot1 - np.random.normal(0, np.sqrt((self._alpha1*(del_rot1**2) + self._alpha2*(del_trans**2))), size=n)
        del_trans_hat = del_trans - np.random.normal(0, np.sqrt((self._alpha3*(del_trans**2) + self._alpha4*(del_rot1**2) + self._alpha4*(del_rot2**2))), size=n)
        del_rot2_hat = del_rot2 - np.random.normal(0, np.sqrt((self._alpha1*(del_rot2**2) + self._alpha2*(del_trans**2))), size=n)
        
        x_updated = x_t0_vector[:,0] + del_trans_hat*np.cos(x_t0_vector[:,2] + del_rot1_hat)
        y_updated = x_t0_vector[:,1] + del_trans_hat*np.sin(x_t0_vector[:,2] + del_rot1_hat)
        theta_updated = x_t0_vector[:,2] + del_rot1_hat + del_rot2_hat
        
        x_t1 = np.array([x_updated, y_updated, theta_updated]).T
        
        return x_t1
        

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]

        JL: see pg 134 of Probabilistic Robotics
        """
        del_rot1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        del_trans = np.sqrt(((u_t0[0] - u_t1[0])**2) + ((u_t0[1] - u_t1[1])**2))
        del_rot2 = u_t1[2] - u_t0[2] - del_rot1
        
        del_rot1_hat = del_rot1 - np.random.normal(0, np.sqrt((self._alpha1*(del_rot1**2) + self._alpha2*(del_trans**2))))
        del_trans_hat = del_trans - np.random.normal(0, np.sqrt((self._alpha3*(del_trans**2) + self._alpha4*(del_rot1**2) + self._alpha4*(del_rot2**2))))
        del_rot2_hat = del_rot2 - np.random.normal(0, np.sqrt((self._alpha1*(del_rot2**2) + self._alpha2*(del_trans**2))))
        
        x_updated = x_t0[0] + del_trans_hat*np.cos(x_t0[2] + del_rot1_hat)
        y_updated = x_t0[1] + del_trans_hat*np.sin(x_t0[2] + del_rot1_hat)
        theta_updated = x_t0[2] + del_rot1_hat + del_rot2_hat
        
        x_t1 = [x_updated, y_updated, theta_updated]
        
        return x_t1
