from polar_route.vessel_performance.vessels.abstract_alr import AbstractALR
from meshiphi.mesh_generation.aggregated_cellbox import AggregatedCellBox
import numpy as np
import logging

class BoatyMcBoatFace(AbstractALR):
    """
        Vessel class with methods specifically designed to model the performance of the ALR BoatyMcBoatFace

        https://projects.noc.ac.uk/oceanids/sites/oceanids/files/images/ALR1500_spec_web_04092020.jpg

        Battery - 95 kWh
        Maximum Duration - 20 days
        Battery Rate - 4.75

    """
    def __init__(self, params):
        """
            Args:
                params (dict): vessel parameters from the vessel config file
        """

        super().__init__(params)
        # Coefficients from fit to figures provided by Alex
        speed_coefficients = np.array([4.44444444, -0.5555555499999991])
        self.speed_polynomial = np.poly1d(speed_coefficients)
        depth_coefficients    = np.array([0.001, 2])
        self.depth_polynomial = np.poly1d(depth_coefficients)

    def model_speed(self, cellbox):
        """
            Method to determine the maximum speed for auto long-range sub can traverse the given cell

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

        battery = [4.75 for s in cellbox.agg_data['speed']]
        cellbox.agg_data['battery'] = battery
        return cellbox
