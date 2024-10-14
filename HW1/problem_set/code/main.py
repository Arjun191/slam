'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
from tqdm import trange


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])

def debug_map(occupancy_map, num_particles):
    plt.rcParams.update({'font.size': 22})
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    mng = plt.get_current_fig_manager()
    plt.ion()
    ax1.imshow(occupancy_map, cmap='Greys')
    ax2.imshow(occupancy_map, cmap='Greys')
    ax3.imshow(occupancy_map, cmap='Greys')
    ax1.set_xlabel("Actual Range Scan ($z_t$)")
    ax1.set_aspect("equal")
    ax2.set_xlabel("Simulated Range Scan ($z_t^*$)")
    ax2.set_aspect("equal")
    ax3.set_xlabel(f"{num_particles} Particles")
    ax3.set_aspect("equal")
    xlim = (300, 700)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax3.set_xlim(xlim)
    ylim = (0, 800)
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)
    ax3.set_ylim(ylim)
    return ax1, ax2, ax3

def debug_timestep(ax1, ax2, ax3, sensor_model, X_bar, z_t, m=180):

    # on ax1 draw the latest laser scan overlayed on the best particle
    angles = np.linspace(0, np.pi, m, endpoint=True)
    best_idx = np.argmax(X_bar[:,3])
    px, py, theta, w = X_bar[best_idx]
    sensor_x = 25 * np.cos(np.degrees(theta)) + px
    sensor_y = 25 * np.sin(np.degrees(theta)) + py

    for ray_angle in angles:
        degrees = (np.degrees(theta) - 90 + np.degrees(ray_angle)) % 360
        dist = z_t[int(np.degrees(ray_angle) % 180)]
        px1 = sensor_x + dist * np.cos(np.radians(degrees))
        py1 = sensor_y + dist * np.sin(np.radians(degrees))
        ax1.plot([sensor_x/10, px1/10], [sensor_y/10, py1/10])

    # on ax2 draw the z_t_star overlayed on the best particle
    angles = np.linspace(0, np.pi, m, endpoint=True)
    best_idx = np.argmax(X_bar[:,3])
    px, py, theta, w = X_bar[best_idx]
    for ray_angle in angles:
        degrees = (np.degrees(theta) - 90 + np.degrees(ray_angle)) % 360
        dist = sensor_model._hash_map[int(sensor_x/10), int(sensor_y/10), int(degrees)]
        px1 = sensor_x + dist * np.cos(np.radians(degrees))
        py1 = sensor_y + dist * np.sin(np.radians(degrees))
        ax2.plot([sensor_x/10, px1/10], [sensor_y/10, py1/10])

    # draw all particles on ax3
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = ax3.scatter(x_locs, y_locs, c=X_bar[:, 3], marker='o', cmap="plasma")

    # plt.show(block=True)
    plt.pause(0.00001)
    # remove all lines and points
    for line in ax1.get_lines():
        line.remove()
    for line in ax2.get_lines():
        line.remove()
    scat.remove()


def visualize_timestep(X_bar, fname, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    if type(fname) == int:
        plt.savefig('{}/{:04d}.png'.format(output_path, fname))
    else:
        plt.savefig('{}/{}.png'.format(output_path, fname))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    DONE : Add your code here
    This version converges faster than init_particles_random
    """
    X_bar_init = np.zeros((num_particles, 4))

    # start with random init
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))

    # mark non-free positions as invalid
    x0_cells = np.floor(x0_vals/10).flatten().astype(int)
    y0_cells = np.floor(y0_vals/10).flatten().astype(int)
    invalid = (occupancy_map[y0_cells, x0_cells] > 1e-6) | (occupancy_map[y0_cells, x0_cells] < -1e-6)
    invalid_count = np.count_nonzero(invalid)
    # print(f"Invalid count {invalid_count}")
    
    # replace invalid points until all are in free space
    while invalid_count:
        y0_vals[invalid] = np.random.uniform(0, 7000, (invalid_count, 1))
        x0_vals[invalid] = np.random.uniform(3000, 7000, (invalid_count, 1))
        x0_cells = np.floor(x0_vals/10).flatten().astype(int)
        y0_cells = np.floor(y0_vals/10).flatten().astype(int)
        invalid = (occupancy_map[y0_cells, x0_cells] > 1e-6) | (occupancy_map[y0_cells, x0_cells] < -1e-6)
        invalid_count = np.count_nonzero(invalid)
        # print(f"Invalid count {invalid_count}")

    # initialize theta and weights for all particles
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_log', default='../data/log/robotdata2.log')
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)
    
    if args.debug:
        ax1, ax2, ax3 = debug_map(occupancy_map, args.num_particles)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        if ((time_stamp <= 0.0) | (meas_type == "O")):
            continue

        # JL: added this. Seems like robot doesn't move for first few seconds,
        # causing us to lose the initial particle diversity
        if (time_stamp <= 5.0):
            continue


        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        
        """
        NON VECTORIZED VERSION
        """
        for i, m in enumerate(range(0, num_particles)):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        X_bar[:,3] = X_bar[:,3]/np.sum(X_bar[:,3])
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize and time_idx %10 ==0:
            visualize_timestep(X_bar, time_idx, args.output)
            
        if args.debug:
            debug_timestep(ax1, ax2, ax3, sensor_model, X_bar, z_t)