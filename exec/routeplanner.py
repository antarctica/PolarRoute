import sys,json
from RoutePlanner.CellGrid import CellGrid
from RoutePlanner.vessel_performance import VesselPerformance
from RoutePlanner.optimisation import TravelTime

# Loading Configuration
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

# Discrete Meshing
mesh = CellGrid(config)

# Vehicle Specs
sf = VesselPerformance(config)

# Route Planning
TT = TravelTime(config)
TT.compute_routes()
TT.compute_smoothed_routes()