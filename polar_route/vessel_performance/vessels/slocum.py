from polar_route.vessel_performance.vessels.abstract_glider import AbstractGlider
from meshiphi.mesh_generation.aggregated_cellbox import AggregatedCellBox
import numpy as np
import logging

class SlocumGlider(AbstractGlider):
    """
        Vessel class with methods specifically designed to model the performance of the Slocum G2 Glider
    """
    def __init__(self, params):
        """
            Args:
                params (dict): vessel parameters from the vessel config file
        """

        super().__init__(params)
        # Coefficients from fit to figures provided by Alex
        speed_coefficients    = np.array([4.44444444, -0.5555555499999991])
        self.speed_polynomial = np.poly1d(speed_coefficients)
        depth_coefficients    = np.array([0.001, 2])
        self.depth_polynomial = np.poly1d(depth_coefficients)

    def model_speed(self, cellbox):
        """
            Method to determine the maximum speed that the glider can traverse the given cell

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with speed values
        """

        cellbox.agg_data['speed'] = [self.max_speed for x in range(8)]
        return cellbox

    def model_battery(self, cellbox):
        """
            Method to determine the rate of battery usage in a given cell in Ah/day

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with battery consumption values
        """

        if cellbox.agg_data['elevation'] > self.max_elevation:
            battery = np.inf
        elif cellbox.agg_data['elevation'] <= -1000.0:
            # Assume yo-yo dive to 1000m
            # Battery consumption changes only with speed
            battery = [self.speed_polynomial(s) for s in cellbox.agg_data['speed']]
        else:
            # Battery consumption change with speed
            battery_speed = [self.speed_polynomial(s) for s in cellbox.agg_data['speed']]

            # Estimate depth variation based on linear fit to figures from Alex's presentation
            battery = [bs * self.depth_polynomial(cellbox.agg_data['elevation']) for bs in battery_speed]

        cellbox.agg_data['battery'] = battery
        return cellbox
