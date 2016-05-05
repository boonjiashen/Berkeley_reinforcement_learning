import math
import random

class SingleDimGrid(object):

    def __init__(self, high=1, low=0, num_intervals=10):
        self.high = high
        self.low = low
        self.num_intervals = num_intervals
        self.interval_width = float(high - low) / num_intervals

    def discretize(self, x):
        ret = int((x - self.low) / self.interval_width)
        ret = max(0, min(self.num_intervals - 1, ret))
        return ret

class MultiDimGrid(object):

    def __init__(self, singleDimGrids):
        """`singleDimGrids` is an array of SingleDimGrid objects.

        The length of the array is the number of dimensions of self.
        """

        self.num_dimensions = len(singleDimGrids)
        self.grid1Ds = singleDimGrids

    def discretize(self, vector):
        return tuple(grid.discretize(x)
                for grid, x in zip(self.grid1Ds, vector))

if __name__ == "__main__":

    import numpy as np
    lims = (0.866, 2.55, .26, 3.2)
    num_intervals = 10
    grids = [SingleDimGrid(high=lim, low=-lim, num_intervals=num_intervals)
            for lim in lims]
    ndgrid = MultiDimGrid(grids)
    for _ in range(10):
        x = [random.gauss(0, 1) for _ in range(4)]
        print(ndgrid.discretize(x), x)

    grid = SingleDimGrid(high=3, low=-2, num_intervals=7)
    for x in np.linspace(-3, 5, 10):
        print(x, grid.discretize(x))
