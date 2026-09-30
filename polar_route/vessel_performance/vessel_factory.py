"""
Vessel factory for model-based vessel performance system.

Creates vessel instances dynamically based on vessel_class specification
in configuration files. Vessels compose resistance and consumption models
defined in configuration.
"""

import logging
from polar_route.vessel_performance.vessels.generic_vessels import Ship, Glider, AUV, Aircraft

logger = logging.getLogger(__name__)


class VesselFactory:
    """
    Factory class to produce initialised vessel objects using pluggable models.
    
    Vessels are instantiated based on 'vessel_class' in configuration:
        - 'ship': Surface vessels with configurable resistance models
        - 'glider': Underwater gliders with battery consumption
        - 'auv': Autonomous underwater vehicles with battery consumption
        - 'aircraft': Airborne vehicles with fuel/battery consumption
    """
    
    # Mapping of vessel_class strings to implementation classes
    VESSEL_CLASSES = {
        "ship": Ship,
        "glider": Glider,
        "auv": AUV,
        "aircraft": Aircraft
    }
    
    @classmethod
    def get_vessel(cls, config):
        """
        Create vessel instance from configuration.
        
        The configuration should be validated against vessel_schema.py before
        calling this method. This factory relies on schema validation for
        parameter checking and model type validation.
        
        Args:
            config (dict): Validated vessel configuration dictionary containing:
                - vessel_class (str): Type of vessel ("ship", "glider", "auv", "aircraft")
                - max_speed (float): Maximum vessel speed
                - unit (str): Speed unit
                - resistance_models (list, optional): Resistance model specifications
                - consumption_model (dict): Consumption model specification
                - Additional vessel-specific parameters
        
        Returns:
            AbstractVessel: Initialized vessel instance with configured models
            
        Raises:
            ValueError: If vessel_class is not recognised
        """
        vessel_class = config.get('vessel_class')
        
        if vessel_class not in cls.VESSEL_CLASSES:
            available = ", ".join(cls.VESSEL_CLASSES.keys())
            raise ValueError(
                f"Unknown vessel_class '{vessel_class}'. "
                f"Available classes: {available}"
            )
        
        vessel_impl = cls.VESSEL_CLASSES[vessel_class]
        logger.info(f"Creating vessel of class '{vessel_class}'")
        
        # Instantiate vessel (will create models via ModelRegistry)
        vessel = vessel_impl(config)
        
        return vessel
