
import unittest
import json
from polar_route.mesh_generation.environment_mesh import EnvironmentMesh
from PolarRoute.polar_route.mesh_builder import MeshBuilder
from polar_route.mesh_generation.neighbour_graph import NeighbourGraph

from polar_route.mesh_generation.boundary import Boundary
class TestEnvMesh(unittest.TestCase):
   def setUp(self):
      self.config = None
      self.env_mesh = None
      self.json_file = "../regression_tests/example_meshes/Enviromental_Meshes/create_mesh.output2013_4_80_new_format.json"
      with open (self.json_file , "r") as config_file:
          self.config = json.load(config_file) ['config']
          self.env_mesh = MeshBuilder(self.config).build_environmental_mesh()
      self.loaded_env_mesh = EnvironmentMesh.load_from_json(self.json_file)
      # self.loaded_env_mesh.save("loaded_mesh.json")
      


   def test_load_from_json (self):
     

      self.assertEqual (self.loaded_env_mesh.bounds.get_bounds() , self.env_mesh.bounds.get_bounds())

      self.assertEqual (len (self.loaded_env_mesh.agg_cellboxes), len (self.env_mesh.agg_cellboxes))
      self.assertEqual (len (self.loaded_env_mesh.neighbour_graph.get_graph()), len (self.env_mesh.neighbour_graph.get_graph()))
   
   def test_update_agg_cellbox(self):
      self.loaded_env_mesh.update_cellbox(0 , {"x":"5"})
      self.assertEqual (self.loaded_env_mesh.agg_cellboxes[0].get_agg_data()["x"] , "5")
   





