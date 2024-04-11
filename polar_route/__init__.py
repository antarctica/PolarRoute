__version__ = "0.4.0"
__description__ = "PolarRoute: Long-distance maritime polar route planning taking into account complex changing environmental conditions"
__license__ = "MIT"
__author__ = "Jonathan Smith, Samuel Hall, George Coombs, Ayat Fekry, Harrison Abbot - Software Developers\n Samuel Hall - Technical Lead\n Maria Fox - Project Manager\n Jonathan Smith - Codebase Champion"
__email__ = "amop@bas.ac.uk"
__copyright__ = "2022-, BAS AI Lab"

# Wrapped in try-except so that setup.py can import polar_route without crashing due to dependency errors
try:
    from meshiphi.mesh_generation.mesh_builder import MeshBuilder as MeshBuilder
    from meshiphi.dataloaders.factory import DataLoaderFactory as DataLoaderFactory
    from meshiphi.mesh_generation.boundary import Boundary as Boundary

    from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller as VesselPerformanceModeller

    from polar_route.route_planner import RoutePlanner as RoutePlanner

except ModuleNotFoundError as err:
    print(f'{err}\n Is PolarRoute installed correctly?')
