
import unittest
import json
from polar_route.route_planner import RoutePlanner
from polar_route.mesh_generation.mesh_builder import MeshBuilder
from polar_route.mesh_generation.neighbour_graph import NeighbourGraph

from polar_route.mesh_generation.boundary import Boundary
class TestEnvMesh(unittest.TestCase):
   def setUp(self):
      self.config = None
      self.mesh_file = "../regression_tests/example_meshes/vessel_meshes/grf_reprojection.json"
      self.wp_file = "./resources/waypoint/waypoints_20160114.csv"
      self.route_conf = "./resources/waypoint/route_config.json"
      self.route_planner= None
      with open (self.route_conf , "r") as config_file:
          self.config = json.load(config_file)
      self.route_planner= RoutePlanner (self.mesh_file, self.route_conf)


   def test_is_valid_wp (self):
      src_wps, dest_wps = self.route_planner._load_waypoints (self.wp_file)
      print (src_wps)
      src_wps=  self.route_planner._validate_wps (src_wps)
      dest_wps =  self.route_planner._validate_wps (dest_wps)
      self.assertEqual (len (src_wps) , 0) # in valid wp so the len should be 0
      self.assertEqual (len (dest_wps) , 0)


   def test_dijkstra(self):
      pass

   
   def test_dijkstra_path(self):

      pass

   def test_compute_route(self):

         pass
   
 

if __name__ == '__main__':
    unittest.main()




