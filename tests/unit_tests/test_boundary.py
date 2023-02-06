
import unittest
import json

from polar_route.Boundary import Boundary
class TestBoundary (unittest.TestCase):
   def setUp(self):
         self.boundary = Boundary([-85,-84.9], [-135,-134.9], ['1970-01-01','2021-12-31'])

   def test_getcx (self):
      center = -135 + ((-134.9 +135)/2)
      self.assertEqual ( center , self.boundary.getcx())

   def test_getcy (self):
      center = -85+ ((-84.9 +85)/2)
      self.assertEqual ( center , self.boundary.getcy())

   def test_get_height (self):
      height = -84.9 +85
      self.assertEqual ( height , self.boundary.get_height())

   def test_get_width (self):
      
      width = -134.9 +135
      self.assertEqual ( width , self.boundary.get_width())

   def test_valid_bounds (self):
    self.assertRaises(ValueError, Boundary , [] , [-135,-134.9], ['1970-01-01','2021-12-31'])
    self.assertRaises(ValueError, Boundary , [3,2] , [-135,-134.9], ['1970-01-01','2021-12-31'])
    self.assertRaises(ValueError , Boundary ,[-85,-84.9] , [3,2], ['1970-01-01','2021-12-31'])
    self.assertRaises(ValueError ,  Boundary ,[-85,-84.9] ,[-135,-134.9], ['2021-12-31', '1970-01-01'])


   def test_get_bounds (self):
    bounds = [[-135, -85], [-135, -84.9], [-134.9, -84.9], [-134.9, -85], [-135, -85]]
    self.assertEqual ( bounds, self.boundary.get_bounds())


   def test_load_from_json (self):
      # /home/aykry/BAS/code/refactored/PolarRoute/tests/regression_tests/example_meshes/Enviromental_Meshes/create_mesh.output2013_4_80_new_format.json
      json_file = "../regression_tests/example_meshes/Enviromental_Meshes/create_mesh.output2013_4_80_new_format.json"
      with open (json_file , "r") as config_file:
          config = json.load(config_file) ['config']
          boundary = Boundary.from_json (config)
          print (boundary.get_bounds())


