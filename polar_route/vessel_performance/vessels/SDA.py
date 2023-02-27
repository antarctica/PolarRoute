from polar_route.AggregatedCellBox import AggregatedCellBox
from polar_route.vessel_performance.vessels.AbstractShip import AbstractShip
import logging

class SDA(AbstractShip):
    """
        Vessel class with methods specifically designed to model the performance of the British Antarctic Survey
        research and supply ship, the RRS Sir David Attenborough (SDA)
    """

    def model_speed(self, cellbox):
        """
            Method to determine the maximum speed that the SDA can traverse the given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                speed (float): the maximum speed that the SDA can traverse the given cell
        """
        pass

    def model_fuel(self, cellbox):
        """
            Method to determine the fuel consumption rate of the SDA in a given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                fuel (float): the rate of fuel consumption for the SDA when traversing the input cell
        """
        pass

    def model_resistance(self, cellbox):
        """
            Method to determine the resistance force acting on the SDA in a given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                resistance (float): the resistance force acting on the SDA when traversing the input cell
        """
        pass

    def invert_resistance(self):
        """
            Method to determine the speed that reduces the resistance force on the SDA to an acceptable value
        """
        pass




