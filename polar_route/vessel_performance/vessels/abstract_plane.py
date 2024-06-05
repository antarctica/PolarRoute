from polar_route.vessel_performance.abstract_vessel import AbstractVessel
from meshiphi.mesh_generation.aggregated_cellbox import AggregatedCellBox
from abc import abstractmethod
import logging


class AbstractPlane(AbstractVessel):
    """
        Abstract class to model the performance of a plane
    """
    def __init__(self, params):
        """
            Args:
                params (dict): vessel parameters from the vessel config file
        """
        self.vessel_params = params
        logging.info(f"Initialising a vessel object of type: {self.vessel_params['vessel_type']}")
        self.max_speed      = self.vessel_params['max_speed']
        self.speed_unit     = self.vessel_params['unit']
        self.max_elevation  = self.vessel_params['max_elevation']
        self.max_ice        = self.vessel_params['max_ice_conc']
        self.excluded_zones = self.vessel_params.get('excluded_zones')


    def model_performance(self, cellbox):
        """
            Method to determine the performance characteristics for a plane

            Args:
                    cellbox (AggregatedCellBox): input cell from environmental mesh
        """
        logging.debug(
            f"Modelling performance in cell {cellbox.id} for a vessel of type: {self.vessel_params['vessel_type']}")
        perf_cellbox = self.model_speed(cellbox)
        perf_cellbox = self.model_fuel(perf_cellbox)

        performance_values = {k: v for k, v in perf_cellbox.agg_data.items() if k not in cellbox.agg_data}

        return performance_values

    def model_accessibility(self, cellbox):
        """
            Method to determine if a given cell is accessible to the plane

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh
            Returns:
                access_values (dict): boolean values for the modelled accessibility criteria
        """
        logging.debug(f"Modelling accessibility in cell {cellbox.id} for a vessel of type: "
                      f"{self.vessel_params['vessel_type']}")
        access_values = dict()

        # Exclude cells due to land or ice
        
        access_values['elevation_max'] = self.elevation_max(cellbox)

        # Exclude any other cell types specified in config
        if self.excluded_zones is not None:
            for zone in self.excluded_zones:
                access_values[zone] = cellbox.agg_data[zone]

        access_values['inaccessible'] = any(access_values.values())
        access_values['land']         = self.land(cellbox)

        return access_values

    def land(self, cellbox):
        """
            Method to determine if a cell is land based on sea level

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

    def elevation_max(self, cellbox):
        """
            Method to determine if the altitude in a cell is too high for a plane based on configured maximum elevation

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh
            Returns:
                elevation_max (bool): boolean that is True if the cell is too elevation_max for a glider
        """
        elevation_max = False
        return elevation_max

    @abstractmethod
    def model_speed(self, cellbox: AggregatedCellBox):
        """
            Method to determine the maximum speed that the plane can traverse the given cell

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with speed values
        """
        raise NotImplementedError

    @abstractmethod
    def model_fuel(self, cellbox: AggregatedCellBox):
        """
            Method to determine the fuel consumption rate of the plane in a given cell

            Args:
                cellbox (AggregatedCellBox): input cell from environmental mesh

            Returns:
                cellbox (AggregatedCellBox): updated cell with battery consumption values
        """
        raise NotImplementedError


