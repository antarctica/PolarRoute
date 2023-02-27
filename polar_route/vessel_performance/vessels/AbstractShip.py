from polar_route.vessel_performance.AbstractVessel import AbstractVessel
from polar_route.AggregatedCellBox import AggregatedCellBox
from abc import abstractmethod
import logging

class AbstractShip(AbstractVessel):
    """
        Abstract class to define the methods and attributes common to any vessel that is a ship
    """
    def __init__(self, params):
        """
            Args:
                params (dict): vessel parameters from the vessel config file
        """
        self.vessel_params = params
        logging.info(f"Initialising a vessel object of type: {self.vessel_params['vessel_type']}")

    def model_performance(self, cellbox):
        """
            Method to determine the performance characteristics for the ship
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                performance_values (dict): the value of the modelled performance characteristics for the ship
        """
        performance_values = dict()

        performance_values['speed'] = self.model_speed(cellbox)
        performance_values['fuel'] = self.model_fuel(cellbox)

        return performance_values

    def model_accessibility(self, cellbox):
        """
            Method to determine if a given cell is accessible to the ship
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                access_values (dict): boolean values for the modelled accessibility criteria
        """
        access_values = dict()

        access_values['land'] = self.land(cellbox)
        access_values['ext_ice'] = self.extreme_ice(cellbox)

        access_values['inaccessible'] = any(access_values.values())

        return access_values

    @abstractmethod
    def model_speed(self, cellbox: AggregatedCellBox):
        """
            Method to determine the maximum speed that the ship can traverse the given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                speed (float): the maximum speed that the ship can traverse the given cell
        """
        raise NotImplementedError

    @abstractmethod
    def model_fuel(self, cellbox: AggregatedCellBox):
        """
            Method to determine the fuel consumption rate of the ship in a given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                fuel (float): the rate of fuel consumption for a ship traversing the input cell
        """
        raise NotImplementedError

    def land(self, cellbox):
        """
            Method to determine if a cell is land based on configured minimum depth.
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh
            Returns:
                land (bool): boolean that is True if the cell is inaccessible due to land
        """
        if 'elevation' not in cellbox.agg_data:
            logging.warning(f"No elevation data in cell {cellbox.id}, cannot determine if it is land")
            land = False
        else:
            land = cellbox.agg_data['elevation'] > self.vessel_params['MinDepth']

        return land

    def extreme_ice(self, cellbox):
        """
            Method to determine if a cell is inaccessible based on configured max ice concentration.
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh
            Returns:
                ext_ice (bool): boolean that is True if the cell is inaccessible due to ice
        """
        if 'SIC' not in self.mesh_df:
            logging.debug(f"No sea ice concentration data in cell {cellbox.id}")
            ext_ice = False
        else:
            ext_ice = cellbox.agg_data['SIC'] > self.vessel_params['MaxIceExtent']

        return ext_ice

    @abstractmethod
    def model_resistance(self, cellbox: AggregatedCellBox):
        """
            Method to determine the resistance force acting on the ship in a given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                resistance (float): the resistance force acting on a ship traversing the input cell
        """
        pass

    @abstractmethod
    def invert_resistance(self):
        """
            Method to determine the speed that reduces the resistance force on the ship to an acceptable value
        """
        pass
