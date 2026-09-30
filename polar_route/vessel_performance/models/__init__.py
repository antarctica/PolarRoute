"""
Model registry infrastructure for pluggable vessel performance models.

This module provides a registry system that allows resistance and consumption models
to self-register and be instantiated dynamically from configuration files.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ResistanceModel(ABC):
    """
    Abstract base class for resistance models.
    
    Resistance models calculate forces opposing vessel motion (ice, wind, waves, etc.)
    and optionally compute safe speeds when resistance exceeds vessel capabilities.
    """
    
    @abstractmethod
    def calculate_resistance(self, cellbox, speed: float, direction: float) -> float:
        """
        Calculate resistance force for a vessel traveling at given speed and direction.
        
        Args:
            cellbox: AggregatedCellBox containing environmental data
            speed (float): Vessel speed in km/h
            direction (float): Heading angle in radians (0 = North, π/2 = East)
            
        Returns:
            float: Resistance force in Newtons (N)
        """
        pass
    
    def invert_resistance(self, cellbox, force_limit: float) -> float:
        """
        Calculate maximum safe speed given a resistance force limit.
        
        Args:
            cellbox: AggregatedCellBox containing environmental data
            force_limit (float): Maximum allowable resistance force in Newtons (N)
            
        Returns:
            float: Maximum safe speed in km/h, or None if not applicable for this model
        """
        return None


class ConsumptionModel(ABC):
    """
    Abstract base class for consumption models.
    
    Consumption models calculate fuel or battery consumption rates based on
    vessel speed and resistance/environmental conditions.
    """
    
    @abstractmethod
    def calculate_consumption(self, speed: float, resistance: float = 0.0, **kwargs) -> float:
        """
        Calculate fuel or battery consumption rate.
        
        Args:
            speed (float): Vessel speed in km/h
            resistance (float): Total resistance force in Newtons (N)
            **kwargs: Additional model-specific parameters (e.g., depth for gliders)
            
        Returns:
            float: Consumption rate (tons/day for fuel, Ah/day or Watts for battery)
        """
        pass


# Global model registry
_MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the global registry.
    
    Usage:
        @register_model("ice_resistance")
        class IceResistance(ResistanceModel):
            ...
    
    Args:
        name (str): Unique identifier for the model (used in config files)
    """
    def decorator(cls):
        if name in _MODEL_REGISTRY:
            logger.warning(f"Model '{name}' already registered, overwriting")
        _MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered model: {name} -> {cls.__name__}")
        return cls
    return decorator


class ModelRegistry:
    """
    Factory for creating model instances from configuration.
    """
    
    @staticmethod
    def create(model_name: str, params: Dict[str, Any]):
        """
        Instantiate a model by name with given parameters.
        
        Args:
            model_name (str): Name of the registered model
            params (dict): Parameters to pass to model constructor
            
        Returns:
            Model instance (ResistanceModel or ConsumptionModel)
            
        Raises:
            ValueError: If model_name is not registered
        """
        if model_name not in _MODEL_REGISTRY:
            available = ", ".join(_MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type '{model_name}'. "
                f"Available models: {available}"
            )
        
        model_class = _MODEL_REGISTRY[model_name]
        logger.info(f"Creating model '{model_name}' with params: {params}")
        return model_class(**params)
    
    @staticmethod
    def get_registered_models() -> List[str]:
        """
        Get list of all registered model names.
        
        Returns:
            list: Sorted list of registered model names
        """
        return sorted(_MODEL_REGISTRY.keys())
    
    @staticmethod
    def is_registered(model_name: str) -> bool:
        """
        Check if a model name is registered.
        
        Args:
            model_name (str): Model name to check
            
        Returns:
            bool: True if model is registered
        """
        return model_name in _MODEL_REGISTRY


# Import model implementations to trigger registration
# (models must be imported after registry infrastructure is defined)
from polar_route.vessel_performance.models import resistance, consumption

# Expose key classes for documentation and imports
__all__ = [
    'ResistanceModel',
    'ConsumptionModel',
    'ModelRegistry',
    'register_model',
]
