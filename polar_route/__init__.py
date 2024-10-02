__version__ = "1.0.0"
__description__ = "PolarRoute: Long-distance maritime polar route planning taking into account complex changing environmental conditions"
__license__ = "MIT"
__author__ = "Autonomous Marine Operations Planning (AMOP) Team, AI Lab, British Antarctic Survey"
__email__ = "amop@bas.ac.uk"
__copyright__ = "2021-, BAS AI Lab"

# Wrapped in try-except so that setup.py can import polar_route without crashing due to dependency errors
try:
    from meshiphi.mesh_generation.mesh_builder import MeshBuilder as MeshBuilder
    from meshiphi.dataloaders.factory import DataLoaderFactory as DataLoaderFactory
    from meshiphi.mesh_generation.boundary import Boundary as Boundary

    from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller as VesselPerformanceModeller
    from polar_route.route_planner.route_planner import RoutePlanner as RoutePlanner

except ModuleNotFoundError as err:
    print(f'{err}\n Is PolarRoute installed correctly?')
