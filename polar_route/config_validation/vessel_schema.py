"""
Vessel configuration schema for model-based vessel performance system.

This schema validates the top-level structure of vessel configuration files that
specify vessel class, resistance models, consumption models, and physical
parameters with units.

Per-model parameters (passed through as ``params``) are validated by
``validate_vessel_config``, which builds each model via the registry at
config-validation time. See the model reference documentation for the full
parameter list and units.
"""

vessel_schema = {
    "type": "object",
    "required": ["vessel_class", "max_speed", "unit", "consumption_model"],
    "additionalProperties": False,
    "properties": {
        "vessel_class": {
            "type": "string",
            "enum": ["ship", "glider", "auv", "aircraft"],
            "description": "Type of vessel: ship (surface), glider/auv (underwater), aircraft (airborne)"
        },
        "max_speed": {
            "type": "number",
            "minimum": 0,
            "description": "Maximum vessel speed [km/h]"
        },
        "unit": {
            "type": "string",
            "description": "Speed unit (should be 'km/hr')"
        },
        "max_ice_conc": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": "Maximum ice concentration accessible [%]"
        },
        "min_depth": {
            "type": "number",
            "minimum": 0,
            "description": "Minimum water depth required [m]"
        },
        "max_altitude": {
            "type": "number",
            "minimum": 0,
            "description": "Maximum altitude for aircraft [m]"
        },
        "max_wave": {
            "type": "number",
            "minimum": 0,
            "description": "Maximum wave height accessible [m]"
        },
        "num_directions": {
            "type": "integer",
            "minimum": 4,
            "default": 8,
            "description": "Number of compass directions to model (e.g., 4, 8, 16, 32). More directions increase computational cost."
        },
        "resistance_models": {
            "type": "array",
            "description": "List of resistance models to apply (forces sum linearly)",
            "items": {
                "type": "object",
                "required": ["type", "params"],
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Registered resistance model name (e.g., 'ice_resistance', 'wind_resistance', 'wave_resistance')"
                    },
                    "params": {
                        "type": "object",
                        "description": "Model-specific parameters with units (see model documentation)"
                    }
                }
            }
        },
        "consumption_model": {
            "type": "object",
            "required": ["type", "params"],
            "description": "Fuel or battery consumption model",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Registered consumption model name (e.g., 'polynomial_fuel', 'polynomial_battery', 'constant_consumption')"
                },
                "params": {
                    "type": "object",
                    "description": "Model-specific parameters with units (see model documentation)"
                }
            }
        },
        "speed_adjustment": {
            "type": "object",
            "description": "Optional wind-based speed adjustment thresholds (for SDA_wind-style behaviour)",
            "properties": {
                "wind_threshold_low": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Lower wind resistance threshold as fraction of force_limit [dimensionless]"
                },
                "wind_threshold_high": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Upper wind resistance threshold as fraction of force_limit [dimensionless]"
                },
                "speed_factor_low": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Speed reduction factor for high wind resistance [dimensionless]"
                },
                "tailwind_factor": {
                    "type": "number",
                    "minimum": 1,
                    "description": "Multiplier for tailwind speed boost calculation [dimensionless]"
                }
            }
        },
        "excluded_zones": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of zone names to exclude from routing"
        },
        "neighbour_splitting": {
            "type": "boolean",
            "description": "Enable splitting of cell neighbours"
        }
    }
}
