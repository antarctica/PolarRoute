# Vessel Configuration Gallery

This gallery provides ready-to-use vessel configurations for common vessel types. Copy and modify these configurations for your specific needs.

## Research Icebreakers

### RRS Sir David Attenborough (Standard)

Research icebreaker for polar operations with ice resistance modeling.

```json
{
  "vessel_class": "ship",
  "max_speed": 26.5,
  "unit": "km/hr",
  "max_ice_conc": 80,
  "min_depth": 10,
  "num_directions": 8,
  "resistance_models": [
    {
      "type": "ice_resistance",
      "params": {
        "k": 4.4,
        "b": -0.8267,
        "n": 2.0,
        "beam": 24.0,
        "force_limit": 96634.5,
        "gravity": 9.81
      }
    },
    {
      "type": "wind_resistance",
      "params": {
        "frontal_area": 750.0,
        "air_density": 1.225,
        "interpolation": "linear",
        "angles": [0, 30, 60, 90, 120, 150, 180],
        "coefficients": [0.94, 0.77, 0.42, 0.48, 0.17, -0.30, -0.34]
      }
    }
  ],
  "consumption_model": {
    "type": "polynomial_fuel",
    "params": {
      "speed_coeffs": [0.00137247, -0.0029601, 0.25290433],
      "resistance_coeffs": [7.75218178e-11, 6.48113363e-06]
    }
  }
}
```

### RRS Sir David Attenborough (Wind Adjustment)

Same vessel with additional speed adjustment for strong wind conditions.

```json
{
  "vessel_class": "ship",
  "max_speed": 26.5,
  "unit": "km/hr",
  "max_ice_conc": 80,
  "min_depth": 10,
  "num_directions": 8,
  "resistance_models": [
    {
      "type": "ice_resistance",
      "params": {
        "k": 4.4,
        "b": -0.8267,
        "n": 2.0,
        "beam": 24.0,
        "force_limit": 96634.5,
        "gravity": 9.81
      }
    },
    {
      "type": "wind_resistance",
      "params": {
        "frontal_area": 750.0,
        "air_density": 1.225,
        "interpolation": "linear",
        "angles": [0, 30, 60, 90, 120, 150, 180],
        "coefficients": [0.94, 0.77, 0.42, 0.48, 0.17, -0.30, -0.34]
      }
    }
  ],
  "consumption_model": {
    "type": "polynomial_fuel",
    "params": {
      "speed_coeffs": [0.00137247, -0.0029601, 0.25290433],
      "resistance_coeffs": [7.75218178e-11, 6.48113363e-06]
    }
  },
  "speed_adjustment": {
    "wind_threshold_low": 0.75,
    "wind_threshold_high": 1.0,
    "speed_factor_low": 0.25,
    "tailwind_factor": 10.0
  }
}
```

## Underwater Gliders

### Slocum Glider

Battery-powered underwater glider.

```json
{
  "vessel_class": "glider",
  "max_speed": 1.25,
  "unit": "km/hr",
  "max_ice_conc": 10,
  "min_depth": 10,
  "num_directions": 8,
  "consumption_model": {
    "type": "polynomial_battery",
    "params": {
      "speed_coeffs": [4.44444444, -0.5555555499999991],
      "depth_coeffs": [0.001, 2]
    }
  }
}
```

### Generic Glider

Simplified glider configuration for shallow-water operations.

```json
{
  "vessel_class": "glider",
  "max_speed": 0.8,
  "unit": "km/hr",
  "max_ice_conc": 80,
  "min_depth": 10,
  "max_depth": 200,
  "num_directions": 8,
  "resistance_models": [],
  "consumption_model": {
    "type": "polynomial_battery",
    "params": {
      "speed_coeffs": [0.0, 0.0, 0.0416],
      "depth_coeffs": [0.0, 0.0]
    }
  }
}
```

## Autonomous Underwater Vehicles

### BoatyMcBoatFace (ALR)

Deep-diving AUV for abyssal research with constant power consumption.

```json
{
  "vessel_class": "auv",
  "max_speed": 3.6,
  "unit": "km/hr",
  "max_ice_conc": 10,
  "min_depth": 10,
  "num_directions": 8,
  "consumption_model": {
    "type": "constant_consumption",
    "params": {
      "consumption_rate": 4.75
    }
  }
}
```

### Generic Deep AUV

High-performance AUV for extreme depths.

```json
{
  "vessel_class": "auv",
  "max_speed": 7.2,
  "unit": "km/hr",
  "max_ice_conc": 100,
  "min_depth": 10,
  "max_depth": 6000,
  "num_directions": 8,
  "resistance_models": [],
  "consumption_model": {
    "type": "constant_consumption",
    "params": {
      "consumption_rate": 0.01
    }
  }
}
```

## Aircraft and UAVs

### Generic Research Aircraft

Fixed-wing aircraft with constant consumption.

```json
{
  "vessel_class": "aircraft",
  "max_speed": 300.0,
  "unit": "km/hr",
  "max_ice_conc": 100,
  "min_depth": 0,
  "num_directions": 16,
  "resistance_models": [],
  "consumption_model": {
    "type": "constant_consumption",
    "params": {
      "consumption_rate": 5.0
    }
  }
}
```

### Long-Endurance UAV

Unmanned aerial vehicle optimised for extended missions.

```json
{
  "vessel_class": "aircraft",
  "max_speed": 120.0,
  "unit": "km/hr",
  "max_ice_conc": 100,
  "min_depth": 0,
  "num_directions": 16,
  "resistance_models": [],
  "consumption_model": {
    "type": "constant_consumption",
    "params": {
      "consumption_rate": 0.5
    }
  }
}
```

## Template Configurations

### Minimal Ship

Bare minimum configuration for a ship without resistance models.

```json
{
  "vessel_class": "ship",
  "max_speed": 20.0,
  "unit": "km/hr",
  "max_ice_conc": 50,
  "min_depth": 5,
  "num_directions": 8,
  "resistance_models": [],
  "consumption_model": {
    "type": "constant_consumption",
    "params": {
      "consumption_rate": 1.0
    }
  }
}
```

### Multi-Model Ship

Example ship using all three resistance models.

```json
{
  "vessel_class": "ship",
  "max_speed": 25.0,
  "unit": "km/hr",
  "max_ice_conc": 80,
  "min_depth": 10,
  "num_directions": 8,
  "resistance_models": [
    {
      "type": "ice_resistance",
      "params": {
        "k": 4.0,
        "b": -0.8,
        "n": 2.0,
        "beam": 20.0,
        "force_limit": 80000.0
      }
    },
    {
      "type": "wind_resistance",
      "params": {
        "frontal_area": 600.0,
        "angles": [0, 45, 90, 135, 180],
        "coefficients": [0.9, 0.6, 0.4, 0.0, -0.3]
      }
    },
    {
      "type": "wave_resistance",
      "params": {
        "beam": 20.0,
        "length": 100.0,
        "c_block": 0.65
      }
    }
  ],
  "consumption_model": {
    "type": "polynomial_fuel",
    "params": {
      "speed_coeffs": [0.0, 0.0, 0.2],
      "resistance_coeffs": [1e-10, 5e-06]
    }
  }
}
```

## Customization Tips

### Adjusting Ice Capability

Modify `k`, `b`, `n` coefficients in `ice_resistance` model:
- **Stronger icebreaker**: Increase `force_limit`, adjust `k` (higher = less resistance)
- **Weaker ice capability**: Decrease `force_limit`, increase `k`

### Tuning Fuel Consumption

Adjust `polynomial_fuel` coefficients:
- **More efficient**: Decrease coefficients (especially speed_coeffs[2])
- **Less efficient**: Increase coefficients

### Adding Wave Resistance

For vessels operating in rough seas, add `wave_resistance`:
```json
{
  "type": "wave_resistance",
  "params": {
    "beam": 24.0,
    "length": 128.0,
    "c_block": 0.7
  }
}
```

### Wind Speed Adjustments

For vessels significantly affected by wind, add `speed_adjustment`:
```json
"speed_adjustment": {
  "wind_threshold_low": 0.75,
  "wind_threshold_high": 1.0,
  "speed_factor_low": 0.25,
  "tailwind_factor": 10.0
}
```

## See Also

- [Vessel Performance Configuration](vessel_performance.md) - Detailed parameter documentation
- [Resistance Models Theory](../methods/resistance_models.md) - Understanding the physics
