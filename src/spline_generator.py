# A class to generate spline curves to reweight events based on kinematic variables.
# Spline curves can be created from graphs (with the graph depending on the event kinematics),
# or from an analytical (lambda) function, function of some kinematic variables.
#
# Created by Lorenzo Giannessi, January 2026
#

import numpy as np
import ROOT

class SplineGenerator:
    def __init__(self, spline_type : str,
                    prior: float = 1.0,
                    power: int = None,
                    range_width: float = None,
                    kin_min: float = None,
                    kin_max: float = None,
                    mirror_point: float = None
                 ):
        """
        Initialize the SplineGenerator with given x and y points.
        """

        self.spline = None

        if (spline_type == "polynomial"):
            if (power is None or range_width is None or kin_min is None or kin_max is None):
                raise ValueError("For polynomial spline type, you must provide: power, range_width, kin_min and kin_max.")
            self.spline = polynomial_spline(power, prior, range_width, kin_min, kin_max)
        elif (spline_type == "mirrored"):
            if (power is None or range_width is None or kin_min is None or kin_max is None or mirror_point is None):
                raise ValueError("For mirrored spline type, you must provide: power, range_width, kin_min, and kin_max.")
            self.spline = mirrored_spline(power, prior, range_width, kin_min, kin_max, mirror_point)

        # I will add more spline types in the future
        else:
            raise ValueError(f"Spline type {spline_type} not recognized.")

    def generate_graph(self, kin_val, kin_key = "kin_key", n_points: int = 10, range_min: float = 0.5, range_max: float = 1.5):
        """
        Generate a spline graph from the provided points.
        Parameters
        ----------
        kin_val : float
            The kinematic variable value to evaluate the spline at.
        n_points : int
            Number of points in the generated graph.
        range_min : float
            Minimum x value for the graph.
        range_max : float
            Maximum x value for the graph.
        Returns
        -------
        ROOT.TGraph
            The generated TGraph, to be used as a spline.
        """
        x_vals = np.linspace(range_min, range_max, n_points)
        y_vals = [self.spline.evaluate(x, kin_val) for x in x_vals]

        g = ROOT.TGraph(n_points)
        for i in range(n_points):
            g.SetPoint(i, x_vals[i], y_vals[i])
        g.SetTitle(f"{kin_key}={kin_val}")
        g.SetName(f"spline_{kin_key}")
        return g

    def empty_graph(self):
        """
        Generate an empty graph (placeholder for NO spline).
        Returns
        -------
        ROOT.TGraph
            The generated TGraph, to be used as a spline.
        """
        g = ROOT.TGraph(1)
        g.SetPoint(0, 1, 1)
        g.SetTitle("No Spline")
        return g


class polynomial_spline:
    def __init__(self, power, prior, range_width, kin_min, kin_max):
        self.power = power
        self.prior = prior
        self.range_width = range_width
        self.kin_min = kin_min
        self.kin_max = kin_max

    def __call__(self, x, kin_val):
        if self.kin_max <= self.kin_min:
            raise ValueError("kin_max and kin_min cannot be equal.")

        return (
            1
            + self.range_width
            * (2 * (kin_val - self.kin_min) / (self.kin_max - self.kin_min) - 1)
            * (x**self.power - self.prior**self.power)
        )

    def evaluate(self, x, kin_val):
        if kin_val > self.kin_max:
            val = self.__call__(x, self.kin_max)
            if val < 0:
                return 0
            else:
                return val
        elif kin_val < self.kin_min:
            val = self.__call__(x, self.kin_min)
            if val < 0:
                return 0
            else:
                return val
        val = self.__call__(x, kin_val)
        if val < 0:
            return 0
        else:
            return val

class mirrored_spline:
    def __init__(self, power, prior, range_width, kin_min, kin_max, mirror_point):
        self.power = power
        self.prior = prior
        self.range_width = range_width
        self.kin_min = kin_min
        self.kin_max = kin_max
        self.mirror_point = mirror_point


    def __call__(self, x, kin_val):
        if self.kin_max <= self.kin_min:
            raise ValueError("kin_max and kin_min cannot be equal.")

        # if x>prior just use polynomial, else mirror around prior
        if x > self.mirror_point:
            poly = polynomial_spline(self.power, self.prior, self.range_width, self.kin_min, self.kin_max)
            return poly.__call__(x, kin_val)
        else:
            poly = polynomial_spline(self.power, self.prior, self.range_width, self.kin_min, self.kin_max)
            mirrored_x = 2*self.mirror_point - x
            return poly.__call__(mirrored_x, kin_val)

    def evaluate(self, x, kin_val):
        if kin_val > self.kin_max:
            val = self.__call__(x, self.kin_max)
            if val < 0:
                return 0
            else:
                return val
        elif kin_val < self.kin_min:
            val = self.__call__(x, self.kin_min)
            if val < 0:
                return 0
            else:
                return val
        val = self.__call__(x, kin_val)
        if val < 0:
            return 0
        else:
            return val