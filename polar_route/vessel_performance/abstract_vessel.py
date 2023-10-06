from meshiphi.mesh_generation.aggregated_cellbox import AggregatedCellBox
from abc import ABCMeta, abstractmethod

class AbstractVessel(metaclass=ABCMeta):
    """
    Interface to define the abstract methods required for any vessel class to work within the VesselPerformanceModeller.
    """

    @abstractmethod
    def __init__(self, params: dict):
        """
        Initialise the vessel object with parameters from the config.

        Args:
            params (dict): vessel parameters from the vessel config file
        """
        raise NotImplementedError

    @abstractmethod
    def model_performance(self, cellbox: AggregatedCellBox):
        """
        Calculate performance parameters for the given vessel.

        Args:
            cellbox (AggregatedCellBox): cell in which performance is being modelled

        Returns:
            performance_values (dict): values for relevant performance parameters
        """
        raise NotImplementedError

    @abstractmethod
    def model_accessibility(self, cellbox: AggregatedCellBox):
        """
        Determine accessibility of the input cell for the given vessel.

        Args:
            cellbox (AggregatedCellBox): cell in which accessibility is being determined

        Returns:
            access_values (dict): values for the accessibility and other related booleans
        """
        raise NotImplementedError
