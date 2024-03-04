from polar_route.vessel_performance.vessels.abstract_uav import AbstractUAV
from meshiphi.mesh_generation.aggregated_cellbox import AggregatedCellBox
import numpy as np
import logging

class Windracer(AbstractUAV):
    """
        Vessel class with methods specifically designed to model the performance of the Twin Otter

        https://windracers.com/drones/

        Cruising Speed - 135 km/hr
        Usage    - 350w
       Duration - 12+ flight duraction
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
            Method to determine the rate of fuel usage in a given cell in gallons/day

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with battery consumption values
        """

        battery = [350 for s in cellbox.agg_data['speed']]

        cellbox.agg_data['battery'] = battery
        return cellbox
