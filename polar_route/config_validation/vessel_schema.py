vessel_schema = {
    "type": "object",
    "required": ["vessel_type", "max_speed",
                 "unit", "max_ice_conc",
                 "min_depth"],
    "additionalProperties": True,
    "properties": {
        "vessel_type": {"type": "string"},
        "max_speed": {"type": "number", "minimum": 0},
        "unit": {"type": "string"},
        "max_ice_conc": {"type": "number", "minimum": 0, "maximum": 100},
        "min_depth": {"type": "number", "minimum": 0},
        "max_wave": {"type": "number", "minimum": 0},
        "excluded_zones": {"type": "array",
                           "items":{"type": "string"}},
        "neighbour_splitting": {"type": "boolean"}
    }
}
