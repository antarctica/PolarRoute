from polar_route.vessel_performance.vessels.abstract_glider import AbstractGlider
from polar_route.mesh_generation.aggregated_cellbox import AggregatedCellBox
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

    def model_speed(self):
        """
            Method to determine the maximum speed that the glider can traverse the given cell
        """
        pass

    def model_battery(self):
        """
            Method to determine the power demand on the battery in a given cell in Ah/day
        """
        pass