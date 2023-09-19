
import unittest


from polar_route.mesh_validation.sampler import Sampler
from polar_route.mesh_validation.mesh_validator import MeshValidator

from polar_route.mesh_generation.boundary import Boundary
class TestMeshValidator(unittest.TestCase):
   

   def setUp(self):
     self.mesh_validator = MeshValidator("../regression_tests/example_meshes/abstract_env_meshes/hgrad.json")
     


   def test_sampler(self):
         sampler = Sampler( 2 , 20)
         ranges = [[10,20] , [100,200]]
         mapped_samples = []
         mapped_samples = sampler.generate_samples(ranges)
        
         
         for sample in mapped_samples:
             self.assertLessEqual (sample[0], ranges[0][1])
             self.assertLessEqual (sample[1], ranges[1][1])
             self.assertGreaterEqual (sample[0], ranges[0][0])
             self.assertGreaterEqual (sample[1], ranges[1][0])

   def test_validate_mesh(self):
      distance = self.mesh_validator.validate_mesh()
      print (distance)
      self.assertLess (distance, 0.1)
   


if __name__ == '__main__':
    unittest.main()


