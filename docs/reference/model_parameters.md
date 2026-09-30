# Model Parameters Reference

Complete reference for all parameters used in vessel performance models.

## Resistance Model Parameters

### Ice Resistance (`ice_resistance`)

| Parameter | Type | Units | Required | Default | Range | Description |
|-----------|------|-------|----------|---------|-------|-------------|
| `k` | float | dimensionless | Yes | - | 2.0 - 6.0 | Scaling coefficient for ice resistance. Lower values indicate more efficient ice-breaking hulls. |
| `b` | float | dimensionless | Yes | - | -1.0 to -0.5 | Beam exponent in resistance formula. Negative values mean wider ships have proportionally less resistance. |
| `n` | float | dimensionless | Yes | - | 1.5 - 2.5 | Froude number exponent. Higher values mean resistance increases more rapidly with speed. |
| `beam` | float | m | Yes | - | 10 - 50 | Vessel beam (width). Used in Froude number calculation and resistance scaling. |
| `force_limit` | float | N | Yes | - | 5×10⁴ - 2×10⁵ | Maximum allowable resistance force. When exceeded, vessel speed is reduced for safety. |
| `gravity` | float | m/s² | No | 9.81 | 9.80 - 9.82 | Gravitational acceleration. Can vary slightly with latitude. |
| `rho_water` | float | N/m³ | No | 9807 | 9500 - 10000 | Water density factor (ρ×g). Varies with water salinity and temperature. |

**Example:**
```json
{
  "type": "ice_resistance",
  "params": {
    "k": 4.4,
    "b": -0.8267,
    "n": 2.0,
    "beam": 24.0,
    "force_limit": 96634.5
  }
}
```

### Wind Resistance (`wind_resistance`)

| Parameter | Type | Units | Required | Default | Range | Description |
|-----------|------|-------|----------|---------|-------|-------------|
| `frontal_area` | float | m² | Yes | - | 100 - 2000 | Frontal area of vessel exposed to wind. Includes superstructure above waterline. |
| `air_density` | float | kg/m³ | No | 1.225 | 1.0 - 1.3 | Air density. Varies with altitude, temperature, and humidity. Sea level standard: 1.225. |
| `interpolation` | string | - | No | "linear" | See below | Method for interpolating drag coefficients between angles: `"linear"`, `"cubic"`, or `"spline"`. |
| `angles` | array[float] | degrees | Yes | - | 0 - 180 | Wind angles relative to vessel heading. Must be sorted ascending. 0° = head wind, 180° = tail wind. |
| `coefficients` | array[float] | dimensionless | Yes | - | -0.5 to 1.5 | Drag coefficients corresponding to each angle. Negative values indicate propulsive assistance (tail wind). |

**Example:**
```json
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
```

### Wave Resistance (`wave_resistance`)

| Parameter | Type | Units | Required | Default | Range | Description |
|-----------|------|-------|----------|---------|-------|-------------|
| `beam` | float | m | Yes | - | 10 - 50 | Vessel beam (width). Used in wave pattern calculations. |
| `length` | float | m | Yes | - | 50 - 200 | Vessel length. Determines wave-making characteristics. |
| `c_block` | float | dimensionless | Yes | - | 0.5 - 0.9 | Block coefficient. Ratio of displaced volume to (length × beam × draft). Lower = more streamlined. |
| `rho_water` | float | N/m³ | No | 9807 | 9500 - 10000 | Water density factor (ρ×g). Varies with water salinity and temperature. |

**Example:**
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

## Consumption Model Parameters

### Polynomial Fuel Model (`polynomial_fuel`)

| Parameter | Type | Units | Required | Default | Description |
|-----------|------|-------|----------|---------|-------------|
| `speed_coeffs` | array[float] | tons/day, tons·hr/day·km, tons·hr²/day·km², ... | Yes | - | Polynomial coefficients for speed-dependent fuel consumption. Index i coefficient multiplies (speed)^i. Typically 2-4 coefficients. |
| `resistance_coeffs` | array[float] | tons/day·N, tons/day·N², ... | Yes | - | Polynomial coefficients for resistance-dependent fuel consumption. Index j coefficient multiplies (resistance)^j. Typically 1-3 coefficients. |

**Consumption Formula:**
$$ C_{fuel} = \sum_{i=0}^{n} a_i \cdot v^i + \sum_{j=1}^{m} b_j \cdot R^j $$

where:
- $C_{fuel}$ is fuel consumption [tons/day]
- $v$ is vessel speed [km/hr]
- $R$ is total resistance [N]
- $a_i$ are `speed_coeffs`
- $b_j$ are `resistance_coeffs`

**Example:**
```json
{
  "type": "polynomial_fuel",
  "params": {
    "speed_coeffs": [0.00137247, -0.0029601, 0.25290433],
    "resistance_coeffs": [7.75218178e-11, 6.48113363e-06]
  }
}
```

### Polynomial Battery Model (`polynomial_battery`)

| Parameter | Type | Units | Required | Default | Description |
|-----------|------|-------|----------|---------|-------------|
| `speed_coeffs` | array[float] | 1/day, hr/day·km, hr²/day·km², ... | Yes | - | Polynomial coefficients for speed-dependent battery drain. Index i coefficient multiplies (speed)^i. |
| `depth_coeffs` | array[float] | 1/day·m, 1/day·m², ... | Yes | - | Polynomial coefficients for depth-dependent battery drain. Index j coefficient multiplies (depth)^j. Depth increases energy for buoyancy control. |

**Consumption Formula:**
$$ C_{battery} = \sum_{i=0}^{n} a_i \cdot v^i + \sum_{j=1}^{m} b_j \cdot d^j $$

where:
- $C_{battery}$ is battery consumption [dimensionless, 0-1 scale]
- $v$ is glider speed [km/hr]
- $d$ is depth [m]
- $a_i$ are `speed_coeffs`
- $b_j$ are `depth_coeffs`

**Example:**
```json
{
  "type": "polynomial_battery",
  "params": {
    "speed_coeffs": [0.0, 0.0, 0.0416],
    "depth_coeffs": [0.0, 0.0]
  }
}
```

### Constant Consumption Model (`constant_consumption`)

| Parameter | Type | Units | Required | Default | Description |
|-----------|------|-------|----------|---------|-------------|
| `consumption_rate` | float | tons/day or dimensionless | Yes | - | Fixed consumption rate independent of operating conditions. Units depend on vessel type. |

**Example:**
```json
{
  "type": "constant_consumption",
  "params": {
    "consumption_rate": 0.01
  }
}
```

## Physical Constants

Default values for physical constants used across models. Can be overridden per-model if needed.

| Constant | Symbol | Default | Units | Description |
|----------|--------|---------|-------|-------------|
| Gravitational acceleration | g | 9.81 | m/s² | Standard gravity |
| Air density | ρ_air | 1.225 | kg/m³ | Standard atmosphere |
| Water density factor | ρ_water | 9807 | N/m³ | Seawater |

## See Also

- [Vessel Performance Configuration](../config/vessel_performance.md) - Configuration guide
- [Resistance Models Theory](../methods/resistance_models.md) - Mathematical background
- [Vessel Gallery](../config/vessel_gallery.md) - Example configurations
