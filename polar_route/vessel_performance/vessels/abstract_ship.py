from polar_route.vessel_performance.abstract_vessel import AbstractVessel
from polar_route.mesh_generation.environment_mesh import AggregatedCellBox
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
        logging.info(f"Initialising a vessel object of type: {self.vessel_params['VesselType']}")

        self.max_speed = self.vessel_params['MaxSpeed']
        self.speed_unit = self.vessel_params['Unit']
        self.max_elevation = -1 * self.vessel_params['MinDepth']
        self.max_ice = self.vessel_params['MaxIceConc']

    def model_performance(self, cellbox):
        """
            Method to determine the performance characteristics for the ship
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                performance_values (dict): the value of the modelled performance characteristics for the ship
        """
        logging.debug(f"Modelling performance in cell {cellbox.id} for a vessel of type: {self.vessel_params['VesselType']}")
        # Check if the speed is defined in the input cellbox
        if 'speed' not in cellbox.agg_data:
            logging.debug(f'No speed in cell, assigning default value of {self.max_speed} '
                          f'{self.speed_unit} from config')
            cellbox.agg_data['speed'] = self.max_speed

        perf_cellbox = self.model_speed(cellbox)
        perf_cellbox = self.model_fuel(perf_cellbox)

        performance_values = {k:v for k,v in perf_cellbox.agg_data.items() if k not in cellbox.agg_data}

        return performance_values

    def model_accessibility(self, cellbox):
        """
            Method to determine if a given cell is accessible to the ship
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                access_values (dict): boolean values for the modelled accessibility criteria
        """
        logging.debug(f"Modelling accessibility in cell {cellbox.id} for a vessel of type: {self.vessel_params['VesselType']}")
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
                cellbox (AggregatedCellBox): updated cell with speed values
        """
        raise NotImplementedError

    @abstractmethod
    def model_fuel(self, cellbox: AggregatedCellBox):
        """
            Method to determine the fuel consumption rate of the ship in a given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with fuel consumption values
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
            land = cellbox.agg_data['elevation'] > self.max_elevation

        return land

    def extreme_ice(self, cellbox):
        """
            Method to determine if a cell is inaccessible based on configured max ice concentration.
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh
            Returns:
                ext_ice (bool): boolean that is True if the cell is inaccessible due to ice
        """
        if 'SIC' not in cellbox.agg_data:
            logging.debug(f"No sea ice concentration data in cell {cellbox.id}")
            ext_ice = False
        else:
            ext_ice = cellbox.agg_data['SIC'] > self.max_ice

        return ext_ice

    @abstractmethod
    def model_resistance(self, cellbox: AggregatedCellBox):
        """
            Method to determine the resistance force acting on the ship in a given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with resistance values
        """
        pass

    @abstractmethod
    def invert_resistance(self, cellbox: AggregatedCellBox):
        """
            Method to determine the speed that reduces the resistance force on the ship to an acceptable value
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh
            Returns:
                speed (float): Safe vessel speed in km/h
        """
        pass
