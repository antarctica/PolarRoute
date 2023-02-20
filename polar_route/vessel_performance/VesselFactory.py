from polar_route.vessel_performance.vessels.SDA import SDA
import json

class VesselFactory:
    """
        Factory class to produce initialised vessel objects.
    """
    @classmethod
    def get_vessel(cls, config_path):
        """

        Args:
            config_path (str): a file path pointing to a vessel config json file

        Returns:
            vessel: a vessel class for performance modelling
        """
        vessel_requirements = {"SDA": (SDA, ["MaxSpeed", "Beam", "HullType", "ForceLimit", "MaxIceExtent", "MinDepth"])}

        with open(config_path, 'r') as f:
            config = json.load(f)

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
