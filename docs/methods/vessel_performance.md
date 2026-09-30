# Methods - Vessel Performance

## Overview

The vessel_performance module provides a model-based architecture for calculating vessel performance characteristics across meshed environmental models. The system uses:

1. **ModelRegistry** - Plugin system for resistance and consumption models
2. **Generic Vessel Classes** - Ship, Glider, AUV, Aircraft implementations
3. **VesselPerformanceModeller** - Orchestrates performance calculations
4. **VesselFactory** - Creates vessel instances from configuration

![](../assets/figures/Mesh_Fuel_Speed.jpg)

*The sea ice concentration (a), speed (b) and fuel consumption (c) for the SDA across the Weddell Sea.
The latter two quantities are derived from the former.*

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           VesselPerformanceModeller                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Coordinates performance & accessibility modeling  │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ uses
                       ↓
┌─────────────────────────────────────────────────────────┐
│                  VesselFactory                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Creates vessel instances from configuration       │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ instantiates
                       ↓
┌─────────────────────────────────────────────────────────┐
│            AbstractVessel (base class)                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • model_performance()  • model_accessibility()    │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┬─────────────┐
         │             │             │             │
         ↓             ↓             ↓             ↓
    ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
    │  Ship  │   │ Glider │   │  AUV   │   │Aircraft│
    └────────┘   └────────┘   └────────┘   └────────┘
         │             │             │             │
         │ composes    │ composes    │ composes    │ composes
         ↓             ↓             ↓             ↓
┌─────────────────────────────────────────────────────────┐
│                   ModelRegistry                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ • ResistanceModel   • ConsumptionModel            │  │
│  │ • @register_model decorator                       │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ↓                           ↓
┌──────────────────┐        ┌──────────────────┐
│ ResistanceModels │        │ConsumptionModels │
│ • ice_resistance │        │ • polynomial_fuel│
│ • wind_resistance│        │ • poly_battery   │
│ • wave_resistance│        │ • constant       │
└──────────────────┘        └──────────────────┘
```

## Component Documentation

### Vessel Performance Modeller

::: polar_route.vessel_performance.vessel_performance_modeller.VesselPerformanceModeller
   options:
      merge_init_into_class: true
      members:
      - model_performance
      - model_accessibility
      - to_json

### Vessel Factory

::: polar_route.vessel_performance.vessel_factory.VesselFactory
   options:
      members:
      - get_vessel

### Abstract Vessel

Base class for all vessel types defining the core interface.

::: polar_route.vessel_performance.abstract_vessel.AbstractVessel
   options:
      merge_init_into_class: true
      members:
      - model_performance
      - model_accessibility

### Generic Vessel Classes

Generic implementations using model composition.

#### Ship

::: polar_route.vessel_performance.vessels.Ship
   options:
      merge_init_into_class: true
      members:
      - model_performance
      - model_accessibility

#### Glider

::: polar_route.vessel_performance.vessels.Glider
   options:
      merge_init_into_class: true
      members:
      - model_performance
      - model_accessibility

#### AUV

::: polar_route.vessel_performance.vessels.AUV
   options:
      merge_init_into_class: true
      members:
      - model_performance
      - model_accessibility

#### Aircraft

::: polar_route.vessel_performance.vessels.Aircraft
   options:
      merge_init_into_class: true
      members:
      - model_performance
      - model_accessibility

### Model Registry

Plugin system for registering and creating performance models.

::: polar_route.vessel_performance.models.ModelRegistry
   options:
      members:
      - create
      - get_registered_models

### Resistance Models

Base class and implementations for calculating forces opposing motion.

::: polar_route.vessel_performance.models.ResistanceModel
   options:
      members:
      - calculate_resistance
      - invert_resistance

::: polar_route.vessel_performance.models.resistance.IceResistance
   options:
      merge_init_into_class: true
      members:
      - calculate_resistance
      - invert_resistance

::: polar_route.vessel_performance.models.resistance.WindResistance
   options:
      merge_init_into_class: true
      members:
      - calculate_resistance

::: polar_route.vessel_performance.models.resistance.WaveResistance
   options:
      merge_init_into_class: true
      members:
      - calculate_resistance

### Consumption Models

Base class and implementations for calculating fuel/battery usage.

::: polar_route.vessel_performance.models.ConsumptionModel
   options:
      members:
      - calculate_consumption

::: polar_route.vessel_performance.models.consumption.PolynomialFuelModel
   options:
      merge_init_into_class: true
      members:
      - calculate_consumption

::: polar_route.vessel_performance.models.consumption.PolynomialBatteryModel
   options:
      merge_init_into_class: true
      members:
      - calculate_consumption

::: polar_route.vessel_performance.models.consumption.ConstantConsumptionModel
   options:
      merge_init_into_class: true
      members:
      - calculate_consumption

## See Also

- [Vessel Performance Configuration](../config/vessel_performance.md) - Configuration reference
- [Resistance Models Theory](resistance_models.md) - Mathematical foundations
- [Vessel Gallery](../config/vessel_gallery.md) - Example configurations

