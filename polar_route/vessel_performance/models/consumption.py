"""
Consumption model implementations for vessel performance calculations.

This module contains registered consumption models that calculate fuel or battery
consumption rates based on vessel speed and resistance/environmental conditions.
"""

import numpy as np
import logging
from polar_route.vessel_performance.models import register_model, ConsumptionModel

logger = logging.getLogger(__name__)


@register_model("polynomial_fuel")
class PolynomialFuelModel(ConsumptionModel):
    """
    Polynomial fuel consumption model for ships.
    
    Calculates fuel consumption using polynomial functions of speed and resistance.
    
    Formula:
        fuel = (c_s2 * v² + c_s1 * v + c_s0 + c_r2 * R² + c_r1 * R) * 24
    
    Where:
        v = vessel speed [km/h]
        R = total resistance [N]
        c_s = speed coefficients
        c_r = resistance coefficients
        * 24 converts from tons/hour to tons/day
    
    Parameters:
        speed_coeffs (list): Speed polynomial coefficients [c_s2, c_s1, c_s0]
                            Units: [tons/day per (km/h)², tons/day per km/h, tons/day]
        resistance_coeffs (list): Resistance polynomial coefficients [c_r2, c_r1]
                                 Units: [tons/day per N², tons/day per N]
    """
    
    def __init__(self, speed_coeffs: list, resistance_coeffs: list):
        if len(speed_coeffs) != 3:
            raise ValueError(f"Expected 3 speed coefficients, got {len(speed_coeffs)}")
        if len(resistance_coeffs) != 2:
            raise ValueError(f"Expected 2 resistance coefficients, got {len(resistance_coeffs)}")
        
        self.speed_coeffs = np.array(speed_coeffs)
        self.resistance_coeffs = np.array(resistance_coeffs)
        
        logger.debug(
            f"Initialized PolynomialFuelModel: speed_coeffs={speed_coeffs}, "
            f"resistance_coeffs={resistance_coeffs}"
        )
    
    def calculate_consumption(self, speed: float, resistance: float = 0.0, **kwargs) -> float:
        """
        Calculate fuel consumption rate.
        
        Args:
            speed (float): Vessel speed [km/h]
            resistance (float): Total resistance force [N]
            **kwargs: Ignored
            
        Returns:
            float: Fuel consumption rate [tons/day]
        """
        # Handle negative resistance (e.g., strong tailwind)
        if resistance < 0:
            resistance = 0.0
        
        # Polynomial calculation
        fuel = (
            self.speed_coeffs[0] * speed**2
            + self.speed_coeffs[1] * speed
            + self.speed_coeffs[2]
            + self.resistance_coeffs[0] * resistance**2
            + self.resistance_coeffs[1] * resistance
        ) * 24.0  # Convert to tons/day
        
        return fuel


@register_model("polynomial_battery")
class PolynomialBatteryModel(ConsumptionModel):
    """
    Polynomial battery consumption model for gliders and AUVs.
    
    Calculates battery consumption using polynomial functions of speed and depth.
    
    Formula:
        battery = poly_speed(v) * poly_depth(d)
    
    Where:
        v = vessel speed
        d = operating depth
        poly_speed/poly_depth = numpy polynomial objects
    
    Parameters:
        speed_coeffs (list): Speed polynomial coefficients [c_n, c_n-1, ..., c_1, c_0]
                           Units: [Ah/day per (km/h)^n] or [W per (km/h)^n]
        depth_coeffs (list): Depth polynomial coefficients [c_n, c_n-1, ..., c_1, c_0]
                           Units: [Ah/day per m^n] or [W per m^n]
    """
    
    def __init__(self, speed_coeffs: list, depth_coeffs: list):
        self.speed_coeffs = np.array(speed_coeffs)
        self.depth_coeffs = np.array(depth_coeffs)
        
        # Create polynomial objects
        self.speed_polynomial = np.poly1d(self.speed_coeffs)
        self.depth_polynomial = np.poly1d(self.depth_coeffs)
        
        logger.debug(
            f"Initialized PolynomialBatteryModel: speed_coeffs={speed_coeffs}, "
            f"depth_coeffs={depth_coeffs}"
        )
    
    def calculate_consumption(self, speed: float, resistance: float = 0.0, 
                            depth: float = 0.0, **kwargs) -> float:
        """
        Calculate battery consumption rate.
        
        Args:
            speed (float): Vessel speed [km/h]
            resistance (float): Not used for battery models
            depth (float): Operating depth [m], required kwarg
            **kwargs: Additional parameters (ignored)
            
        Returns:
            float: Battery consumption rate [Ah/day or W]
        """
        battery = self.speed_polynomial(speed) * self.depth_polynomial(depth)
        return battery


@register_model("constant_consumption")
class ConstantConsumptionModel(ConsumptionModel):
    """
    Constant consumption model for aircraft and simple vehicles.
    
    Returns a fixed consumption rate regardless of speed or conditions.
    
    Parameters:
        consumption_rate (float): Fixed consumption rate [tons/day, Ah/day, or W]
    """
    
    def __init__(self, consumption_rate: float):
        self.consumption_rate = consumption_rate
        
        logger.debug(f"Initialized ConstantConsumptionModel: consumption_rate={consumption_rate}")
    
    def calculate_consumption(self, speed: float, resistance: float = 0.0, **kwargs) -> float:
        """
        Calculate consumption rate (constant).
        
        Args:
            speed (float): Vessel speed [km/h] (ignored)
            resistance (float): Total resistance force [N] (ignored)
            **kwargs: Ignored
            
        Returns:
            float: Fixed consumption rate [tons/day, Ah/day, or W]
        """
        return self.consumption_rate
