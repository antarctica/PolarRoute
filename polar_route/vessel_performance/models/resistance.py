"""
Resistance model implementations for vessel performance calculations.

This module contains registered resistance models that calculate forces opposing
vessel motion through ice, wind, and waves.
"""

import numpy as np
import logging
from polar_route.vessel_performance.models import register_model, ResistanceModel

logger = logging.getLogger(__name__)


@register_model("ice_resistance")
class IceResistance(ResistanceModel):
    """
    Froude-based ice resistance model for ships.
    
    Calculates ice resistance force based on vessel speed, ice properties, and
    hull-specific parameters. Uses Froude number formulation.
    
    Parameters:
        k (float): Hull coefficient [dimensionless]
        b (float): Froude number exponent [dimensionless]
        n (float): Ice concentration exponent [dimensionless]
        beam (float): Vessel beam width [m]
        force_limit (float): Maximum allowable resistance force [N]
        gravity (float): Gravitational acceleration [m/s²], default 9.81
    """
    
    def __init__(self, k: float, b: float, n: float, beam: float, 
                 force_limit: float, gravity: float = 9.81):
        self.k = k
        self.b = b
        self.n = n
        self.beam = beam
        self.force_limit = force_limit
        self.gravity = gravity
        
        logger.debug(
            f"Initialized IceResistance: k={k}, b={b}, n={n}, "
            f"beam={beam}m, force_limit={force_limit}N"
        )
    
    def calculate_resistance(self, cellbox, speed: float, direction: float = None) -> float:
        """
        Calculate ice resistance force at given speed.
        
        Formula:
            R_ice = 0.5 * k * Fr^b * ρ * B * h * v² * C^n
        
        Where:
            Fr = Froude number = v / √(g * C * h)
            ρ = ice density [kg/m³]
            B = beam width [m]
            h = ice thickness [m]
            v = vessel speed [m/s]
            C = ice concentration [fraction 0-1]
        
        Args:
            cellbox: AggregatedCellBox with SIC, thickness, density fields
            speed (float): Vessel speed [km/h]
            direction (float): Not used for ice resistance (isotropic)
            
        Returns:
            float: Ice resistance force [N]
        """
        # Extract ice properties
        sic = cellbox.agg_data.get("SIC", 0.0)
        thickness = cellbox.agg_data.get("thickness", 0.0)
        density = cellbox.agg_data.get("density", 0.0)
        
        # No ice = no resistance
        if not sic or not thickness:
            return 0.0
        
        # Convert speed from km/h to m/s
        speed_ms = speed * (5.0 / 18.0)
        
        # Calculate Froude number
        ice_conc_fraction = sic / 100.0  # Convert from percentage to fraction
        froude = speed_ms / np.sqrt(self.gravity * ice_conc_fraction * thickness)
        
        # Calculate resistance
        resistance = (
            0.5 * self.k * (froude ** self.b) * density * self.beam * thickness
            * (speed_ms ** 2) * (ice_conc_fraction ** self.n)
        )
        
        return resistance
    
    def invert_resistance(self, cellbox, force_limit: float = None) -> float:
        """
        Calculate maximum safe speed given resistance force limit.
        
        Solves ice resistance equation for speed when R_ice = force_limit.
        
        Args:
            cellbox: AggregatedCellBox with SIC, thickness, density fields
            force_limit (float): Maximum resistance [N], uses self.force_limit if None
            
        Returns:
            float: Maximum safe speed [km/h]
        """
        if force_limit is None:
            force_limit = self.force_limit
        
        # Extract ice properties
        sic = cellbox.agg_data.get("SIC", 0.0)
        thickness = cellbox.agg_data.get("thickness", 0.0)
        density = cellbox.agg_data.get("density", 0.0)
        
        # No ice = no limit
        if not sic or not thickness:
            return np.inf
        
        ice_conc_fraction = sic / 100.0
        
        # Solve for velocity: v = [analytical expression]^(1/(2+b))
        v_exp = (
            2 * force_limit / (
                self.k * density * self.beam * thickness * (ice_conc_fraction ** self.n)
                * (self.gravity * thickness * ice_conc_fraction) ** -(self.b / 2)
            )
        )
        
        v_ms = v_exp ** (1.0 / (2.0 + self.b))
        v_kmh = v_ms * (18.0 / 5.0)  # Convert m/s to km/h
        
        return v_kmh


@register_model("wind_resistance")
class WindResistance(ResistanceModel):
    """
    Wind resistance model using drag coefficients and apparent wind.
    
    Calculates wind resistance based on apparent wind (true wind minus vessel velocity),
    frontal area, and direction-dependent drag coefficients.
    
    Parameters:
        frontal_area (float): Vessel frontal area exposed to wind [m²]
        angles (list): Wind angle breakpoints [degrees], e.g., [0, 30, 60, 90, 120, 150, 180]
        coefficients (list): Drag coefficients at each angle [dimensionless]
        interpolation (str): Interpolation method ('linear', 'cubic', 'spline'), default 'linear'
        air_density (float): Air density [kg/m³], default 1.225
    """
    
    def __init__(self, frontal_area: float, angles: list, coefficients: list,
                 interpolation: str = "linear", air_density: float = 1.225):
        self.frontal_area = frontal_area
        self.air_density = air_density
        self.interpolation = interpolation
        
        # Convert angles from degrees to radians
        self.angles_rad = np.array(angles) * (np.pi / 180.0)
        self.coefficients = np.array(coefficients)
        
        logger.debug(
            f"Initialized WindResistance: frontal_area={frontal_area}m², "
            f"air_density={air_density}kg/m³, interpolation={interpolation}"
        )
    
    def _get_drag_coefficient(self, rel_angle: float) -> float:
        """
        Get drag coefficient for relative wind angle using interpolation.
        
        Args:
            rel_angle (float): Relative wind angle [radians]
            
        Returns:
            float: Drag coefficient [dimensionless]
        """
        if self.interpolation == "linear":
            return np.interp(rel_angle, self.angles_rad, self.coefficients)
        elif self.interpolation == "cubic":
            from scipy.interpolate import interp1d
            f = interp1d(self.angles_rad, self.coefficients, kind='cubic', 
                        fill_value="extrapolate")
            return float(f(rel_angle))
        elif self.interpolation == "spline":
            from scipy.interpolate import UnivariateSpline
            spl = UnivariateSpline(self.angles_rad, self.coefficients, k=3, s=0)
            return float(spl(rel_angle))
        else:
            logger.warning(f"Unknown interpolation '{self.interpolation}', using linear")
            return np.interp(rel_angle, self.angles_rad, self.coefficients)
    
    def _calculate_apparent_wind(self, cellbox, vessel_speed: float, heading: float):
        """
        Calculate apparent wind speed and direction.
        
        Args:
            cellbox: AggregatedCellBox with u10, v10 wind components
            vessel_speed (float): Vessel speed [km/h]
            heading (float): Vessel heading [radians], 0=North, π/2=East
            
        Returns:
            tuple: (apparent_wind_speed [m/s], relative_angle [radians])
        """
        # Get true wind components (m/s)
        u_wind = cellbox.agg_data.get("u10", 0.0)  # Eastward
        v_wind = cellbox.agg_data.get("v10", 0.0)  # Northward
        
        # Convert vessel speed to m/s
        vessel_speed_ms = vessel_speed * 0.277778
        
        # Calculate vessel velocity components
        u_vessel = vessel_speed_ms * np.sin(heading)  # Eastward
        v_vessel = vessel_speed_ms * np.cos(heading)  # Northward
        
        # Calculate apparent wind (true wind - vessel velocity)
        u_apparent = u_wind - u_vessel
        v_apparent = v_wind - v_vessel
        
        # Calculate apparent wind magnitude
        wind_speed = np.sqrt(u_apparent**2 + v_apparent**2)
        
        # Calculate relative angle between vessel heading and apparent wind
        if wind_speed == 0 or vessel_speed_ms == 0:
            return 0.0, 0.0
        
        # Vessel direction unit vector
        vessel_unit = np.array([u_vessel, v_vessel]) / vessel_speed_ms if vessel_speed_ms > 0 else np.array([0, 0])
        
        # Apparent wind unit vector
        wind_unit = np.array([u_apparent, v_apparent]) / wind_speed if wind_speed > 0 else np.array([0, 0])
        
        # Calculate angle (dot product gives cos of angle between them)
        dot_product = np.dot(vessel_unit, wind_unit)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Numerical safety
        # Use negative to convert from angle between vectors to angle from wind source
        # Wind vector points where wind goes, but drag depends on where it comes from
        relative_angle = np.arccos(-dot_product)
        
        return wind_speed, relative_angle
    
    def calculate_resistance(self, cellbox, speed: float, direction: float) -> float:
        """
        Calculate wind resistance force.
        
        Formula:
            R_wind = 0.5 * ρ * v_apparent² * A * C_d(θ) - 0.5 * ρ * v_vessel² * A * C_d(0)
        
        Args:
            cellbox: AggregatedCellBox with u10, v10 wind components
            speed (float): Vessel speed [km/h]
            direction (float): Vessel heading [radians]
            
        Returns:
            float: Wind resistance force [N]
        """
        # Check if wind data available
        if "u10" not in cellbox.agg_data or "v10" not in cellbox.agg_data:
            return 0.0
        
        # Calculate apparent wind
        wind_speed, rel_angle = self._calculate_apparent_wind(cellbox, speed, direction)
        
        # Get drag coefficients
        c_d_rel = self._get_drag_coefficient(rel_angle)
        c_d_head = self._get_drag_coefficient(0.0)  # Head-on coefficient
        
        # Convert vessel speed to m/s
        vessel_speed_ms = speed * 0.277778
        
        # Calculate resistance
        resistance = (
            0.5 * self.air_density * (wind_speed ** 2) * self.frontal_area * c_d_rel
            - 0.5 * self.air_density * (vessel_speed_ms ** 2) * self.frontal_area * c_d_head
        )
        
        return resistance


@register_model("wave_resistance")
class WaveResistance(ResistanceModel):
    """
    Wave resistance model using Kreitner formula.
    
    Calculates wave resistance based on significant wave height and vessel geometry.
    Valid for small wave heights (< 2m). Recommended by ITTC.
    
    Reference: https://ittc.info/media/1936/75-04-01-012.pdf
    
    Parameters:
        beam (float): Vessel beam width [m]
        length (float): Vessel length [m]
        c_block (float): Block coefficient (ratio of underwater volume to cuboid) [dimensionless], default 0.75
        rho_water (float): Specific weight of water [N/m³], default 9807 (freshwater at 4°C)
    """
    
    def __init__(self, beam: float, length: float, c_block: float = 0.75, 
                 rho_water: float = 9807):
        self.beam = beam
        self.length = length
        self.c_block = c_block
        self.rho_water = rho_water
        
        logger.debug(
            f"Initialized WaveResistance: beam={beam}m, length={length}m, "
            f"c_block={c_block}, rho_water={rho_water}N/m³"
        )
    
    def calculate_resistance(self, cellbox, speed: float, direction: float = None) -> float:
        """
        Calculate wave resistance force using Kreitner formula.
        
        Formula:
            R_wave = (0.64 * ρ_w * c_block * h² * B²) / L
        
        Where:
            ρ_w = specific weight of water [N/m³]
            c_block = block coefficient [dimensionless]
            h = significant wave height [m]
            B = beam width [m]
            L = vessel length [m]
        
        Args:
            cellbox: AggregatedCellBox with swh (significant wave height) field
            speed (float): Vessel speed [km/h] (not used in Kreitner formula)
            direction (float): Not used for wave resistance (assumed isotropic)
            
        Returns:
            float: Wave resistance force [N]
        """
        # Get significant wave height
        wave_height = cellbox.agg_data.get("swh", 0.0)
        
        if wave_height == 0:
            return 0.0
        
        # Kreitner formula (valid up to ~2m wave height)
        resistance = (
            0.64 * self.rho_water * self.c_block * (wave_height ** 2) 
            * (self.beam ** 2)
        ) / self.length
        
        return resistance
