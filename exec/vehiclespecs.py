import sys,json
from RoutePlanner.vessel_performance import VesselPerformance

# Loading Configuration
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

# Vehicle Specs
sf = VesselPerformance(config)