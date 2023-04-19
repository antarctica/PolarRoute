__version__ = "0.1.0"
__description__ = "PolarRoute: Long-distance maritime polar route planning taking into account complex changing environmental conditions"
__license__ = "MIT"
__author__ = "Jonathan Smith, Samuel Hall, George Coombs, Harrison Abbot, Ayat Fekry, James Byrne, Michael Thorne, Maria Fox"
__email__ = "jonsmi@bas.ac.uk"
__copyright__ = "2022, BAS AI Lab"

from polar_route.mesh_generation.mesh_builder import MeshBuilder as MeshBuilder
from polar_route.dataloaders.factory import DataLoaderFactory as DataLoaderFactory

from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller as VesselPerformanceModeller

from polar_route.route_planner import RoutePlanner as RoutePlanner
