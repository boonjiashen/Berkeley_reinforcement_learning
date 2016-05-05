import math
import random

class SingleDimGrid(object):

    def __init__(self, high=1, low=0, num_intervals=10):
        self.high = high
        self.low = low
        self.num_intervals = num_intervals
        self.interval_width = float(high - low) / num_intervals

    def discretize(self, x):
        quanta = int((x - self.low) / self.interval_width)
        quanta = max(0, min(self.num_intervals - 1, quanta))
        return quanta

    def undiscretize(self, quantum):
        return quantum * self.interval_width + self.low

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

    def undiscretize(self, quanta):
        return tuple(grid.undiscretize(quantum)
                for grid, quantum in zip(self.grid1Ds, quanta))

if __name__ == "__main__":

    import numpy as np
    lims = (0.866, 2.55, .26, 3.2)
    lims = (10, 5, 3, 1)
    num_intervals = 10
    grids = [SingleDimGrid(high=lim, low=-lim, num_intervals=num_intervals)
            for lim in lims]
    grids = [SingleDimGrid(high=2, low=-2, num_intervals=4),
            SingleDimGrid(high=-3, low=-6, num_intervals=6),
            ]
    ndgrid = MultiDimGrid(grids)
    for _ in range(10):
        x = [random.gauss(0, 1) for _ in range(len(grids))]
        quanta = ndgrid.discretize(x)
        print(quanta, x, ndgrid.undiscretize(quanta))

    grid = SingleDimGrid(high=3, low=-2, num_intervals=7)
    for x in np.linspace(-3, 5, 10):
        print(x, grid.discretize(x))
