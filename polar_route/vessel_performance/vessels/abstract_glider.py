from polar_route.vessel_performance.abstract_vessel import AbstractVessel
from polar_route.mesh_generation.aggregated_cellbox import AggregatedCellBox
from abc import abstractmethod
import logging


class AbstractGlider(AbstractVessel):
    """
        Class to model the performance of an underwater glider
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
        self.min_depth = self.vessel_params['MinDepth']
        self.max_ice = self.vessel_params['MaxIceConc']


    def model_performance(self, cellbox):
        """
            Method to determine the performance characteristics for the underwater glider
                Args:
                    cellbox (AggregatedCellBox): input cell from environmental mesh
        """
        logging.debug(
            f"Modelling performance in cell {cellbox.id} for a vessel of type: {self.vessel_params['VesselType']}")
        perf_cellbox = self.model_speed(cellbox)
        perf_cellbox = self.model_battery(perf_cellbox)

        performance_values = {k: v for k, v in perf_cellbox.agg_data.items() if k not in cellbox.agg_data}

        return performance_values

    def model_accessibility(self, cellbox):
        """
            Method to determine if a given cell is accessible to the underwater glider
                 Args:
                    cellbox (AggregatedCellBox): input cell from environmental mesh
                Returns:
                    access_values (dict): boolean values for the modelled accessibility criteria
        """
        logging.debug(f"Modelling accessibility in cell {cellbox.id} for a vessel of type: "
                      f"{self.vessel_params['VesselType']}")
        access_values = dict()

        access_values['land'] = self.land(cellbox)
        access_values['shallow'] = self.shallow(cellbox)
        access_values['ext_ice'] = self.extreme_ice(cellbox)

        access_values['inaccessible'] = any(access_values.values())

        return access_values

    def land(self, cellbox):
        """
            Method to determine if a cell is land based on sea level.
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh
            Returns:
                land (bool): boolean that is True if the cell is inaccessible due to land
        """
        if 'elevation' not in cellbox.agg_data:
            logging.warning(f"No elevation data in cell {cellbox.id}, cannot determine if it is land")
            land = False
        else:
            land = cellbox.agg_data['elevation'] >= 0.0

        return land

    def shallow(self, cellbox):
        """
            Method to determine if the water in a cell is too shallow for a glider based on configured minimum depth.
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh
            Returns:
                shallow (bool): boolean that is True if the cell is too shallow for a glider
        """
        if 'elevation' not in cellbox.agg_data:
            logging.warning(f"No elevation data in cell {cellbox.id}, cannot determine if it is too shallow")
            shallow = False
        else:
            shallow = 0.0 > cellbox.agg_data['elevation'] > self.min_depth

        return shallow

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
    def model_speed(self, cellbox: AggregatedCellBox):
        """
            Method to determine the maximum speed that the glider can traverse the given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with speed values
        """
        raise NotImplementedError

    @abstractmethod
    def model_battery(self, cellbox: AggregatedCellBox):
        """
            Method to determine the battery consumption rate of the glider in a given cell
            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with battery consumption values
        """
        raise NotImplementedError


