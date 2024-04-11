from polar_route.vessel_performance.vessels.abstract_plane import AbstractPlane
from meshiphi.mesh_generation.aggregated_cellbox import AggregatedCellBox
import numpy as np
import logging

class TwinOtter(AbstractPlane):
    """
        Vessel class with methods specifically designed to model the performance of the Twin Otter

        https://www.vikingair.com/twin-otter-series-400/technical-description

        Usage    - 3.85 tonnes fuel per day 
        Max Fuel - long-range=1.447,normal=1.15
    """
    def __init__(self, params):
        """
            Args:
                params (dict): vessel parameters from the vessel config file
        """

        super().__init__(params)
        speed_coefficients    = np.array([4.44444444, -0.5555555499999991])
        self.speed_polynomial = np.poly1d(speed_coefficients)
        depth_coefficients    = np.array([0.001, 2])
        self.depth_polynomial = np.poly1d(depth_coefficients)

    def model_speed(self, cellbox):
        """
            Method to determine the maximum speed that the twin-otter can traverse the given cell

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with speed values
        """

        cellbox.agg_data['speed'] = [self.max_speed for x in range(8)]
        return cellbox

    def model_fuel(self, cellbox):
        """
            Method to determine the rate of fuel usage in a given cell in tonnes/day

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with battery consumption values
        """

        fuel = [3.858 for s in cellbox.agg_data['speed']]

        cellbox.agg_data['fuel'] = fuel
        return cellbox
