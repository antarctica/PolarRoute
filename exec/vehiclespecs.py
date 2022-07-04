import sys,json
from RoutePlanner.speed import SpeedFunctions

# Loading Configuration
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

# Vehicle Specs
sf = SpeedFunctions(config)