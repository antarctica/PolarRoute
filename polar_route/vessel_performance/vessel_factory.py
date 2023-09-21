from polar_route.vessel_performance.vessels.SDA import SDA
from polar_route.vessel_performance.vessels.slocum import SlocumGlider

class VesselFactory:
    """
        Factory class to produce initialised vessel objects.
    """
    @classmethod
    def get_vessel(cls, config):
        """
            Method to return an initialised instance of a vessel class designed for performance modelling

            Args:
                config (dict): a vessel config dictionary

            Returns:
                vessel: an instance of a vessel class designed for performance modelling
        """
        vessel_requirements = {"SDA": (SDA, ["max_speed", "unit", "beam", "hull_type", "force_limit", "max_ice_conc",
                                             "min_depth"]),
                               "Slocum": (SlocumGlider, ["max_speed", "unit", "max_ice_conc", "min_depth"])}

        vessel_type = config['vessel_type']

        if vessel_type in vessel_requirements:
            vessel_class = vessel_requirements[vessel_type][0]
            required_params = vessel_requirements[vessel_type][1]
        else:
            raise ValueError(f'{vessel_type} not in known list of vessels')

        assert all(key in config for key in required_params), \
            f'Dataloader {vessel_type} is missing some parameters! Requires {required_params}. Has {list(config.keys())}'

        vessel = vessel_class(config)

        return vessel
