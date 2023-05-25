
import unittest
import json
from polar_route.mesh_generation.environment_mesh import EnvironmentMesh
from polar_route.mesh_generation.mesh_builder import MeshBuilder
from polar_route.mesh_generation.direction import Direction

from polar_route.mesh_generation.boundary import Boundary
class TestMeshBuilder(unittest.TestCase):
   def setUp(self):
      self.config = None
      self.env_mesh = None
      self.json_file = "../unit_tests/resources/global_grf_normal.json"
      with open (self.json_file , "r") as config_file:
          self.json_file = json.load(config_file)
          self.config = self.json_file ['config']
          self.mesh_builder =  MeshBuilder(self.config)
          self.env_mesh = self.mesh_builder.build_environmental_mesh()
          self.env_mesh.save("global_mesh.json")
        
      


   def test_check_global_mesh (self):
      self.assertEqual (self.mesh_builder.neighbour_graph.get_neighbour_case(self.mesh_builder.mesh.cellboxes[0] , self.mesh_builder.mesh.cellboxes[71]) , Direction.west)

    

    




