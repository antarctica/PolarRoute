import sys,json
from RoutePlanner.CellGrid import CellGrid

# Loading Configuration
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

# Discrete Meshing
mesh = CellGrid(config)