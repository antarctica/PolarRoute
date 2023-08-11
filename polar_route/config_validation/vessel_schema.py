vessel_schema = {
    "type": "object",
    "required": ["VesselType", "MaxSpeed",
                 "Unit", "MaxIceConc",
                 "MinDepth"],
    "additionalProperties": True,
    "properties": {
        "VesselType": {"type": "string"},
        "MaxSpeed": {"type": "number", "minimum": 0},
        "Unit": {"type": "string"},
        "MaxIceConc": {"type": "number", "minimum": 0, "maximum": 100},
        "MinDepth": {"type": "number", "minimum": 0}
    }
}
