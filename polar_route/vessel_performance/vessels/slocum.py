from polar_route.vessel_performance.vessels.abstract_glider import AbstractGlider
from polar_route.mesh_generation.aggregated_cellbox import AggregatedCellBox
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

        if cellbox.agg_data['elevation'] > self.min_depth:
            fuel = np.inf
        elif cellbox.agg_data['elevation'] <= -1000.0:
            # Assume yo-yo dive to 1000m
            fuel = 5.0
        else:
            # Estimate based on linear fit to figures from Alex's presentation
            coefficients = np.array([0.005, 1])
            polynomial = np.poly1d(coefficients)
            fuel = polynomial(cellbox.agg_data['elevation'])

        cellbox.agg_data['fuel'] = [fuel for x in range(8)]
        return cellbox