# Resistance Models - Theory and Implementation

This document provides the mathematical theory and implementation details for resistance models used in vessel performance calculations.

## Overview

Resistance models calculate forces opposing vessel motion through ice, water, and air. PolarRoute implements three primary resistance models:

1. **Froude Ice Resistance** - Ice breaking and friction
2. **Wind Drag Resistance** - Aerodynamic resistance
3. **Kreitner Wave Resistance** - Hydrodynamic wave-making resistance

Vessels can use multiple resistance models simultaneously - the total resistance is the sum of all active models.

## Froude Ice Resistance

### Physical Basis

Ice resistance arises from breaking ice ahead of the vessel and overcoming friction as ice pieces slide along the hull. The Froude number-based formulation scales resistance with vessel speed and geometry using dimensionless parameters.

### Mathematical Formulation

The ice resistance force is calculated as:

$$ R_{ice} = k \cdot \text{Fr}^n \cdot \rho_{water} \cdot g \cdot beam^b $$

where the Froude number is:

$$ \text{Fr} = \frac{v}{\sqrt{g \cdot beam}} $$

**Parameters:**
- $R_{ice}$ - Ice resistance force [N]
- $k$ - Dimensionless scaling coefficient (vessel-specific)
- $\text{Fr}$ - Froude number (dimensionless)
- $n$ - Froude number exponent (dimensionless, typically ~2)
- $\rho_{water}$ - Water density factor [N/m³], default 9807
- $g$ - Gravitational acceleration [m/s²], default 9.81
- $beam$ - Vessel beam (width) [m]
- $b$ - Beam exponent (dimensionless, typically negative)
- $v$ - Vessel speed [m/s]

### Force Limit

A maximum resistance force can be specified:

$$ R_{effective} = \min(R_{ice}, R_{limit}) $$

When $R_{ice} > R_{limit}$, the vessel speed is reduced to maintain safe operations.

### Inverse Calculation

To find the safe speed for a given force limit, we invert the resistance equation:

$$ v_{safe} = \left( \frac{R_{limit}}{k \cdot \rho_{water} \cdot g \cdot beam^b} \right)^{1/n} \cdot \sqrt{g \cdot beam} $$

### Parameter Selection

Typical parameter ranges:

| Parameter | Typical Range | Physical Meaning |
|-----------|---------------|------------------|
| $k$ | 2.0 - 6.0 | Hull efficiency (lower = more efficient) |
| $b$ | -1.0 to -0.5 | Beam scaling (negative = wider ships have less resistance per unit width) |
| $n$ | 1.5 - 2.5 | Speed sensitivity (higher = resistance increases faster with speed) |
| $beam$ | 10 - 50 m | Ship width |
| $R_{limit}$ | 50,000 - 200,000 N | Maximum safe resistance |

## Wind Drag Resistance

### Physical Basis

Wind creates aerodynamic drag on the vessel's structure. The drag force depends on:
- Relative wind speed (apparent wind = true wind - vessel motion)
- Wind angle relative to heading
- Vessel frontal area exposed to wind
- Angle-dependent drag coefficient

### Mathematical Formulation

The wind drag force is:

$$ R_{wind} = \frac{1}{2} \rho_{air} \cdot A_{frontal} \cdot C_d(\theta) \cdot v_{apparent}^2 $$

where:

$$ v_{apparent} = \sqrt{v_{wind}^2 + v_{vessel}^2 - 2 v_{wind} v_{vessel} \cos(\theta)} $$

**Parameters:**
- $R_{wind}$ - Wind resistance force [N]
- $\rho_{air}$ - Air density [kg/m³], default 1.225
- $A_{frontal}$ - Frontal area [m²]
- $C_d(\theta)$ - Drag coefficient (function of wind angle)
- $\theta$ - Wind angle relative to vessel heading [degrees]
- $v_{apparent}$ - Apparent wind speed [m/s]
- $v_{wind}$ - True wind speed [m/s]
- $v_{vessel}$ - Vessel speed [m/s]

### Drag Coefficient Interpolation

The drag coefficient varies with wind angle. Users provide discrete angle-coefficient pairs, and the model interpolates between them using:
- **Linear** interpolation (default) - Simple, fast
- **Cubic** interpolation - Smoother curves
- **Spline** interpolation - Maximum smoothness

### Wind Angle Convention

- 0° = Head wind (wind opposes motion)
- 90° = Beam wind (perpendicular)
- 180° = Tail wind (wind assists motion)

Negative drag coefficients for tail winds indicate propulsive assistance.

## Kreitner Wave Resistance

### Physical Basis

As vessels move through water, they generate waves that propagate away, carrying energy. This wave-making resistance becomes significant at higher speeds and for vessels with poor hydrodynamic efficiency.

### Mathematical Formulation

The Kreitner wave resistance formulation:

$$ R_{wave} = f(v, beam, length, C_B, \rho_{water}) $$

**Parameters:**
- $R_{wave}$ - Wave resistance force [N]
- $beam$ - Vessel beam (width) [m]
- $length$ - Vessel length [m]
- $C_B$ - Block coefficient (dimensionless, 0.5-0.9)
- $\rho_{water}$ - Water density factor [N/m³], default 9807
- $v$ - Vessel speed [m/s]

### Block Coefficient

The block coefficient $C_B$ represents hull fullness:

$$ C_B = \frac{V_{displaced}}{length \times beam \times draft} $$

### Speed Regimes

Wave resistance exhibits different behaviour in different speed regimes:
- **Low speed** ($\text{Fr}_L < 0.3$): Negligible
- **Moderate speed** ($0.3 < \text{Fr}_L < 0.5$): Significant and increasing
- **High speed** ($\text{Fr}_L > 0.5$): Dominant resistance component

where $\text{Fr}_L = \frac{v}{\sqrt{g \cdot length}}$ is the length-based Froude number.

## Model Combination

### Total Resistance

When multiple resistance models are active, the total resistance is:

$$ R_{total} = R_{ice} + R_{wind} + R_{wave} + \ldots $$

Each model calculates resistance independently, and the sum determines the vessel's achievable speed.

### Speed Calculation

Given total resistance $R_{total}$, the vessel speed is determined by the power-speed relationship encoded in the consumption model. For polynomial fuel consumption:

$$ v_{achievable} = f^{-1}(R_{total}, P_{available}) $$

For a worked multi-model vessel configuration, see the
[Vessel Gallery](../config/vessel_gallery.md).

## See Also

- [Vessel Performance Configuration](../config/vessel_performance.md) - Parameter reference
- [Model Parameters Reference](../reference/model_parameters.md) - Complete parameter listing
- [Vessel Gallery](../config/vessel_gallery.md) - Example configurations
