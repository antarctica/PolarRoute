from polar_route.vessel_performance.abstract_vessel import AbstractVessel
from polar_route.mesh_generation.aggregated_cellbox import AggregatedCellBox


class UnderwaterVessel(AbstractVessel):
    """
        Class to model the performance of an underwater vessel
    """

    def __init__(self, params):
        """
            Args:
                params (dict): vessel parameters from the vessel config file
        """
        self.vessel_params = params

    def model_performance(self, cellbox):
        """
            Method to determine the performance characteristics for the underwater vessel
                Args:
                    cellbox (AggregatedCellBox): input cell from environmental mesh
        """
        pass

    def model_accessibility(self, cellbox):
        """
            Method to determine if a given cell is accessible to the underwater vessel
                 Args:
                    cellbox (AggregatedCellBox): input cell from environmental mesh
        """
        pass

