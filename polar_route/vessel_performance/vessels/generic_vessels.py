"""
Generic vessel class implementations using pluggable models.

This module contains generic vessel classes (Ship, Glider, AUV, Aircraft) that
compose resistance and consumption models from configuration files.
"""

import numpy as np
import logging
from polar_route.vessel_performance.abstract_vessel import AbstractVessel
from polar_route.vessel_performance.models import ModelRegistry

logger = logging.getLogger(__name__)


class Ship(AbstractVessel):
    """
    Generic ship vessel with configurable resistance and consumption models.
    
    Ships can be affected by ice, wind, and wave resistance.
    Performance is calculated by summing resistance forces from configured models
    and computing fuel consumption based on total resistance.
    
    Configuration parameters:
        max_speed (float): Maximum vessel speed [km/h]
        unit (str): Speed unit (should be "km/hr")
        max_ice_conc (float): Maximum ice concentration accessible [%]
        min_depth (float): Minimum water depth required [m]
        num_directions (int): Number of compass directions to model (default 8)
        resistance_models (list): List of resistance model specs with 'type' and 'params'
        consumption_model (dict): Consumption model spec with 'type' and 'params'
        speed_adjustment (dict, optional): Speed adjustment thresholds for wind effects
    """
    
    def __init__(self, params: dict):
        """
        Initialize ship with configured models.
        
        Args:
            params (dict): Configuration dictionary
        """
        self.vessel_params = params
        
        # Basic vessel properties
        self.max_speed = params["max_speed"]
        self.unit = params["unit"]
        self.max_ice = params.get("max_ice_conc", 100.0)
        self.min_depth = params.get("min_depth", 0.0)
        self.num_directions = params.get("num_directions", 8)
        self.max_wave = params.get("max_wave", None)
        self.excluded_zones = params.get("excluded_zones", None)
        
        # Speed adjustment parameters (for wind-based speed reduction)
        self.speed_adjustment = params.get("speed_adjustment", None)
        
        # Initialize resistance models
        self.resistance_models = []
        for model_spec in params.get("resistance_models", []):
            model = ModelRegistry.create(model_spec["type"], model_spec["params"])
            self.resistance_models.append(model)
            logger.info(f"Added resistance model: {model_spec['type']}")
        
        # Initialize consumption model
        consumption_spec = params["consumption_model"]
        self.consumption_model = ModelRegistry.create(
            consumption_spec["type"], consumption_spec["params"]
        )
        logger.info(f"Initialized consumption model: {consumption_spec['type']}")
        
        logger.info(
            f"Initialized Ship: max_speed={self.max_speed}{self.unit}, "
            f"num_directions={self.num_directions}, "
            f"{len(self.resistance_models)} resistance models"
        )
    
    def _get_direction_headings(self):
        """
        Calculate heading angles for configured number of directions.
        
        Returns:
            list: Heading angles in radians, starting from NE and going clockwise
        """
        # Standard pattern: start at NE (π/4) and go clockwise
        step = 2 * np.pi / self.num_directions
        headings = [np.pi / 4 + i * step for i in range(self.num_directions)]
        # Normalise last heading to 0 (North) if it's close
        if abs(headings[-1] - 2 * np.pi) < 0.01:
            headings[-1] = 0.0
        return headings
    
    def _calculate_resistance(self, cellbox, speed: float, direction: float) -> float:
        """
        Calculate total resistance by summing all configured resistance models.
        
        Args:
            cellbox: AggregatedCellBox with environmental data
            speed (float): Vessel speed [km/h]
            direction (float): Heading angle [radians]
            
        Returns:
            float: Total resistance force [N]
        """
        total_resistance = 0.0
        
        for model in self.resistance_models:
            resistance = model.calculate_resistance(cellbox, speed, direction)
            total_resistance += resistance
        
        return total_resistance
    
    def _apply_speed_adjustment(self, speed: float, wind_resistance: float) -> float:
        """
        Apply speed adjustment based on wind resistance (legacy SDA_wind logic).
        
        Args:
            speed (float): Base speed [km/h]
            wind_resistance (float): Wind resistance force [N]
            
        Returns:
            float: Adjusted speed [km/h]
        """
        if self.speed_adjustment is None:
            return speed
        
        # Extract thresholds and factors from config
        force_limit = self.vessel_params.get("force_limit", np.inf)
        threshold_low = self.speed_adjustment.get("wind_threshold_low", 0.75)
        threshold_high = self.speed_adjustment.get("wind_threshold_high", 1.0)
        factor_low = self.speed_adjustment.get("speed_factor_low", 0.25)
        tailwind_factor = self.speed_adjustment.get("tailwind_factor", 10.0)
        
        # Headwind logic
        if wind_resistance > 0:
            if wind_resistance < force_limit * threshold_low:
                # Moderate headwind: reduce speed proportionally
                speed = speed * (1 - wind_resistance / force_limit)
            else:
                # Strong headwind: severe speed reduction
                speed = speed * factor_low
        else:
            # Tailwind: small speed boost
            speed = speed * (1 - wind_resistance / (tailwind_factor * force_limit))
        
        return speed
    
    def model_performance(self, cellbox):
        """
        Calculate vessel performance (speed, resistance, fuel) for the cell.
        
        Returns dict with new values to be added to cellbox.agg_data:
            - speed: list of speeds for each direction [km/h]
            - resistance: list of resistances for each direction [N]
            - fuel: list of fuel consumption rates for each direction [tons/day]
        
        Args:
            cellbox: AggregatedCellBox
            
        Returns:
            dict: Performance values to add to cellbox
        """
        logger.debug(f"Calculating performance for cell {cellbox.id}")
        
        # Initialize with default speed
        base_speed = cellbox.agg_data.get("speed", self.max_speed)
        
        # Check ice accessibility and adjust speed if needed
        ice_resistance = None
        if all(k in cellbox.agg_data for k in ("SIC", "thickness", "density")):
            sic = cellbox.agg_data.get("SIC", 0.0)
            
            if sic == 0.0:
                # No ice: use max speed
                base_speed = self.max_speed
                ice_resistance = 0.0
            elif sic > self.max_ice:
                # Too much ice: inaccessible
                base_speed = 0.0
                ice_resistance = np.inf
            else:
                # Calculate ice resistance at max speed
                ice_model = next((m for m in self.resistance_models 
                                if hasattr(m, 'force_limit')), None)
                if ice_model:
                    test_resistance = ice_model.calculate_resistance(
                        cellbox, self.max_speed, 0.0
                    )
                    if test_resistance > ice_model.force_limit:
                        # Ice resistance too high: reduce speed
                        base_speed = ice_model.invert_resistance(cellbox)
                        ice_resistance = ice_model.calculate_resistance(
                            cellbox, base_speed, 0.0
                        )
                    else:
                        base_speed = self.max_speed
                        ice_resistance = test_resistance
        
        # Calculate performance for each direction
        headings = self._get_direction_headings()
        speeds = []
        resistances = []
        fuels = []
        
        # Store wind resistance separately for speed adjustment
        wind_resistances = [0.0] * self.num_directions
        relative_wind_speeds = [0.0] * self.num_directions
        relative_wind_angles = [0.0] * self.num_directions
        
        for i, heading in enumerate(headings):
            speed = base_speed
            
            # Calculate total resistance at this heading
            resistance = self._calculate_resistance(cellbox, speed, heading)
            
            # Extract wind resistance component and wind data if wind model exists
            for model in self.resistance_models:
                if model.__class__.__name__ == "WindResistance":
                    wind_resistances[i] = model.calculate_resistance(
                        cellbox, speed, heading
                    )
                    # Also get apparent wind speed and angle
                    relative_wind_speeds[i], relative_wind_angles[i] = model._calculate_apparent_wind(
                        cellbox, speed, heading
                    )
            
            # Apply speed adjustment if configured
            if self.speed_adjustment:
                speed = self._apply_speed_adjustment(speed, wind_resistances[i])
                # Recalculate resistance at adjusted speed
                resistance = self._calculate_resistance(cellbox, speed, heading)
            
            # Calculate fuel consumption
            fuel = self.consumption_model.calculate_consumption(speed, resistance)
            
            speeds.append(speed)
            resistances.append(resistance)
            fuels.append(fuel)
        
        # Build return dictionary
        performance = {
            "speed": speeds,
            "resistance": resistances,
            "fuel": fuels
        }
        
        # Add ice resistance if calculated
        if ice_resistance is not None:
            performance["ice resistance"] = ice_resistance
        
        # Add wind data if available
        if any(wr != 0 for wr in wind_resistances):
            performance["wind resistance"] = wind_resistances
            performance["relative wind speed"] = relative_wind_speeds
            performance["relative wind angle"] = relative_wind_angles
        
        return performance
    
    def model_accessibility(self, cellbox):
        """
        Determine if cell is accessible based on ice concentration and water depth.
        
        Returns dict with boolean flags:
            - land: True if elevation > -min_depth (above configured max elevation)
            - ext_ice: True if ice concentration > max_ice_conc
            - ext_waves: True if wave height > max_wave (if configured)
            - inaccessible: True if cell cannot be traversed
        
        Args:
            cellbox: AggregatedCellBox
            
        Returns:
            dict: Accessibility flags
        """
        access = {}

        # Check if on land (elevation > -min_depth, matching legacy max_elevation check)
        if 'elevation' not in cellbox.agg_data:
            logger.warning(f"No elevation data in cell {cellbox.id}, cannot determine if it is land")
            access["land"] = False
        else:
            access["land"] = cellbox.agg_data['elevation'] > -self.min_depth

        # Check ice concentration
        if 'SIC' not in cellbox.agg_data or cellbox.agg_data['SIC'] is None:
            access["ext_ice"] = False
        else:
            access["ext_ice"] = cellbox.agg_data['SIC'] > self.max_ice

        # Check wave height if configured
        if self.max_wave is not None:
            if 'swh' not in cellbox.agg_data:
                access["ext_waves"] = False
            else:
                access["ext_waves"] = cellbox.agg_data['swh'] > self.max_wave

        # Check excluded zones if configured
        if self.excluded_zones is not None:
            for zone in self.excluded_zones:
                try:
                    access[zone] = cellbox.agg_data[zone]
                except KeyError:
                    logger.debug(f'{zone} not found in agg cellbox!')

        access["inaccessible"] = any(access.values())

        return access


class AUV(AbstractVessel):
    """
    Generic Autonomous Underwater Vehicle with configurable consumption model.
    
    AUVs are battery-powered underwater vehicles.
    Performance depends on speed and depth polynomials.
    
    Configuration parameters:
        max_speed (float): Maximum speed [km/h]
        unit (str): Speed unit
        max_ice_conc (float): Maximum ice concentration accessible [%]
        min_depth (float): Minimum water depth required [m]
        num_directions (int): Number of compass directions (default 8)
        consumption_model (dict): Battery consumption model with 'type' and 'params'
    """

    #: Whether operating depth is capped at max_dive_depth (yo-yo dive assumption)
    caps_dive_depth = False

    def __init__(self, params: dict):
        self.vessel_params = params
        self.max_speed = params["max_speed"]
        self.unit = params["unit"]
        self.max_ice = params.get("max_ice_conc", 100.0)
        self.min_depth = params.get("min_depth", 0.0)
        self.num_directions = params.get("num_directions", 8)
        
        # Initialize consumption model
        consumption_spec = params["consumption_model"]
        self.consumption_model = ModelRegistry.create(
            consumption_spec["type"], consumption_spec["params"]
        )
        
        logger.info(f"Initialized {type(self).__name__}: max_speed={self.max_speed}{self.unit}")
    
    def model_performance(self, cellbox):
        """Calculate performance (speed and battery consumption)."""
        speed = cellbox.agg_data.get("speed", self.max_speed)
        speeds = [speed] * self.num_directions
        
        # Get elevation (negative value, depth below sea level)
        elevation = cellbox.agg_data.get("elevation", 0.0)
        
        # Cap depth at max dive depth (yo-yo dive assumption) for gliders
        if self.caps_dive_depth:
            max_dive_depth = self.vessel_params.get("max_dive_depth", 1000.0)
            elevation = max(elevation, -max_dive_depth)
        
        # Calculate battery consumption
        battery = self.consumption_model.calculate_consumption(
            speed, depth=elevation
        )
        batteries = [battery] * self.num_directions
        
        return {
            "speed": speeds,
            "battery": batteries
        }
    
    def model_accessibility(self, cellbox):
        """Determine underwater-vehicle accessibility."""
        access = {
            "land": False,
            "ext_ice": False,
            "shallow": False,
            "inaccessible": False
        }
        
        elevation = cellbox.agg_data.get("elevation", -100)
        if elevation >= 0:
            access["land"] = True
            access["inaccessible"] = True
            return access
        
        depth = abs(elevation)
        if depth < self.min_depth:
            access["shallow"] = True
            access["inaccessible"] = True
            return access
        
        sic = cellbox.agg_data.get("SIC", 0.0)
        if sic > self.max_ice:
            access["ext_ice"] = True
            access["inaccessible"] = True
        
        return access


class Glider(AUV):
    """
    Generic underwater glider with configurable consumption model.
    
    A glider is a type of AUV that caps its operating depth at a maximum dive
    depth (yo-yo dive assumption). Otherwise it behaves like an AUV
    (battery-powered, same accessibility rules).
    
    Configuration: Same as AUV
    """
    
    caps_dive_depth = True


class Aircraft(AbstractVessel):
    """
    Generic aircraft with configurable consumption model.
    
    Aircraft operate above surface and are not affected by ice or water depth.
    
    Configuration parameters:
        max_speed (float): Maximum aircraft speed [km/h]
        unit (str): Speed unit
        max_ice_conc (float): Maximum ice concentration (usually ignored)
        num_directions (int): Number of compass directions (default 8)
        consumption_model (dict): Fuel consumption model (usually constant)
    """
    
    def __init__(self, params: dict):
        self.vessel_params = params
        self.max_speed = params["max_speed"]
        self.unit = params["unit"]
        self.num_directions = params.get("num_directions", 8)
        self.max_altitude = params.get("max_altitude", None)
        
        consumption_spec = params["consumption_model"]
        self.consumption_model = ModelRegistry.create(
            consumption_spec["type"], consumption_spec["params"]
        )
        
        logger.info(f"Initialized Aircraft: max_speed={self.max_speed}{self.unit}")
    
    def model_performance(self, cellbox):
        """Calculate aircraft performance (constant speed and fuel)."""
        speed = self.max_speed
        speeds = [speed] * self.num_directions
        
        fuel = self.consumption_model.calculate_consumption(speed)
        fuels = [fuel] * self.num_directions
        
        return {
            "speed": speeds,
            "fuel": fuels
        }
    
    def model_accessibility(self, cellbox):
        """Aircraft can access most cells but check altitude limits."""
        access = {}

        # Check if elevation is too high
        if self.max_altitude is not None:
            if 'elevation' not in cellbox.agg_data:
                access["elevation_max"] = False
            else:
                access["elevation_max"] = cellbox.agg_data['elevation'] > self.max_altitude

        # Check excluded zones if configured
        excluded_zones = self.vessel_params.get('excluded_zones')
        if excluded_zones is not None:
            for zone in excluded_zones:
                try:
                    access[zone] = cellbox.agg_data[zone]
                except KeyError:
                    logger.debug(f'{zone} not found in agg cellbox!')

        access["inaccessible"] = any(access.values())

        # Land flag is informational only (doesn't affect aircraft accessibility)
        if 'elevation' not in cellbox.agg_data:
            access["land"] = False
        else:
            access["land"] = cellbox.agg_data['elevation'] >= 0.0

        return access
