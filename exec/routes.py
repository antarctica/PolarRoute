import sys,json
from RoutePlanner.optimisation import TravelTime

# Loading Configuration
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

# Route Planning
TT = TravelTime(config)
TT.compute_routes()
TT.compute_smoothed_routes()