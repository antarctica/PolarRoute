
import unittest
import json

from polar_route.mesh_generation.mesh_builder import MeshBuilder
class TestJGrid (unittest.TestCase):
   def setUp(self):

      self.conf = None
      with open("resources/feb_2013_Jgrid_config.json", "r") as config_file:
         self.conf = json.load(config_file)['config']



   def test_jgrid(self):
         mesh_builder = MeshBuilder(self.conf)
         env_mesh = mesh_builder.build_environmental_mesh()
   
         env_mesh.save ("resources/jgrid_output.json")
