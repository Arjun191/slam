'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import math
import numpy as np
from tqdm import tqdm 

from matplotlib import pyplot as plt
from matplotlib import figure as fig


class MapReader:
    def __init__(self, src_path_map):

        self._occupancy_map = np.genfromtxt(src_path_map, skip_header=7)
        self._occupancy_map[self._occupancy_map < 0] = -1
        # The raw data stores P(free) the probability a cell is freespace
        # Convert to P(occupancy) by 1-P(free)
        self._occupancy_map[self._occupancy_map > 0] = 1 - self._occupancy_map[
            self._occupancy_map > 0]
        self._occupancy_map = np.flipud(self._occupancy_map)

        self._resolution = 10  # each cell has a 10cm resolution in x,y axes
        self._size_x = self._occupancy_map.shape[0] * self._resolution
        self._size_y = self._occupancy_map.shape[1] * self._resolution

        print('Finished reading 2D map of size: ({}, {})'.format(
            self._size_x, self._size_y))

        # JL: added param, free if less than free_thres
        self._free_thres = 0.35

    def visualize_map(self):
        fig = plt.figure()
        plt.ion()
        plt.imshow(self._occupancy_map, cmap='Greys')
        plt.axis([0, 800, 0, 800])
        plt.draw()
        plt.savefig('map.png')
        plt.pause(0)
        plt.close(fig)

    def get_map(self):
        return self._occupancy_map

    def get_map_size_x(self):  # in cm
        return self._size_x

    def get_map_size_y(self):  # in cm
        return self._size_y
    
    def precompute_rays(self):
        hash_map = np.zeros((800, 800, 360))
        for i in tqdm(range(800), desc="loop over rows"):
            for j in range(800):
                thetas = np.linspace(0, 2*np.pi, 360, endpoint=False)
                px = map1.cell_to_point(i)
                py = map1.cell_to_point(j)
                hash_map[i, j] = self.simulate_scan(px, py, thetas)

        np.save('hash_map.npy', hash_map)

    def test_simulate_scan(self, px, py): 
        thetas = np.linspace(0, 2*np.pi, 360, endpoint=False)
        # dist = self.simulate_scan(px, py, thetas)
        dist = self.test_hash_map(px, py, thetas)
        for i, theta in enumerate(thetas):
            px1 = px + dist[i] * np.cos(theta)
            py1 = py + dist[i] * np.sin(theta)
            plt.plot([px/self._resolution, px1/self._resolution], 
                     [py/self._resolution, py1/self._resolution])

    def simulate_scan(self, px, py, thetas):
        dist = np.zeros(len(thetas))
        for i, theta in enumerate(thetas):
            dist[i] = self.ray_cast(px, py, theta)
        return dist
    
    def test_hash_map(self, px, py, thetas):
        hash_map = np.load('../data/map/hash_map_v2.npy')
        dist = np.zeros(len(thetas))
        for i, theta in enumerate(thetas):
            dist[i] = hash_map[int(px/10), int(py/10), int(np.degrees(theta))]
        return dist

    def cell_in_grid(self, cx, cy):
        in_x = (0 <= cx * self._resolution < self.get_map_size_x())
        in_y = (0 <= cy * self._resolution < self.get_map_size_y())
        return in_x and in_y

    def is_free(self, cx, cy):
        return self._occupancy_map[cy, cx] < self._free_thres

    def point_to_cell(self, p):
        return int((p - 0.5*self._resolution) / self._resolution)

    def cell_to_point(self, c):
        return c * self._resolution + 0.5 * self._resolution

    def ray_cast(self, px, py, rad):
        """
        # Amanatides Woo ray casting algorithm
        params:
            px = starting point, float
            py = starting point, float
            rad = direction in radians, float
            res = resolution (units per cell), float
        returns:
            distance (cm, float)
        notation:
        c = cell coordinates (indices, int)
        p = point coordinates (cm, float)
        """
        cx = self.point_to_cell(px)
        cy = self.point_to_cell(py)

        # determine step direction in x and y
        dx = np.cos(rad)
        dy = np.sin(rad)
        if dy > 0.0:
            step_row = 1
        else:
            step_row = -1
        if dx > 0.0:
            step_col = 1
        else:
            step_col = -1

        if abs(dx) > 1e-6:
            tDeltaX = step_col * (self._resolution/dx)
        else:
            tDeltaX = np.inf
        if abs(dy) > 1e-6:
            tDeltaY = step_row * (self._resolution/dy)
        else:
            tDeltaY = np.inf

        px_lower = self.cell_to_point(cx) - 0.5*self._resolution
        py_lower = self.cell_to_point(cy) - 0.5*self._resolution
        px_upper = px_lower + 0.5*self._resolution
        py_upper = py_lower + 0.5*self._resolution
        tMaxX, tMaxY = np.inf, np.inf
        if dx > 0.0:
            tMaxX = (px_upper - px) / dx
        elif dx < 0.0:
            tMaxX = (px_lower - px) / dx
        if dy > 0.0:
            tMaxY = (py_upper - py) / dy
        elif dy < 0.0:
            tMaxY = (py_lower - py) / dy

        # traverse cell-by-cell
        cx_cur = cx
        cy_cur = cy
        cx_next = cx_cur
        cy_next = cy_cur
        while self.is_free(cx_cur, cy_cur):
            if tMaxX < tMaxY:
                cx_next += step_col
                tMaxX += tDeltaX
            else:
                cy_next += step_row
                tMaxY += tDeltaY

            if self.cell_in_grid(cx_next, cy_next):
                cx_cur = cx_next
                cy_cur = cy_next
            else:
                break

        distx = self.cell_to_point(cx_cur) - px
        disty = self.cell_to_point(cy_cur) - py
        dist = math.sqrt(distx**2 + disty**2)
        return dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    map1 = MapReader(src_path_map)
    print(map1.get_map_size_x(),map1.get_map_size_y())
    
    # map1.precompute_rays()

    # test ray casting at a single point
    cx, cy = (413, 580)
    px = map1.cell_to_point(cx)
    py = map1.cell_to_point(cy)
    map1.test_simulate_scan(px, py)

    cx, cy = (600, 150)
    px = map1.cell_to_point(cx)
    py = map1.cell_to_point(cy)
    map1.test_simulate_scan(px, py)
    plt.imshow(map1._occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])
    plt.show()