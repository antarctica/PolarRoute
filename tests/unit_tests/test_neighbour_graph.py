
import unittest
from polar_route.Direction import Direction
from polar_route.NeighbourGraph import NeighbourGraph
from polar_route.Metadata import Metadata
from polar_route.DataLoader import DataLoaderFactory
from polar_route.cellbox import CellBox

from polar_route.Boundary import Boundary
class TestNeighbourGraph (unittest.TestCase):
   def setUp(self):
         boundary = Boundary([-85,-84.9], [-135,-134.9], ['1970-01-01','2021-12-31'])
         cellbox = CellBox (boundary , 1)
         params = {
      'file': './datastore/bathymetry/GEBCO/gebco_2022_n-40.0_s-90.0_w-140.0_e0.0.nc',
		'downsample_factors': (5,5),
		'data_name': 'elevation',
		'aggregate_type': 'MAX',
       'value_fill_types': "parent"
         }
         split_conds = {
	'threshold': 620,
	'upper_bound': 0.9,
	'lower_bound': 0.1
	}
         factory = DataLoaderFactory
         gebco = factory.get_dataloader('GEBCO', params, min_dp = 5)
         cellbox.set_data_source ([Metadata (gebco , [split_conds] , params ['aggregate_type'] , params ['value_fill_types'])])
         self.cellboxes = self.cellbox.split()
         cell_width = 2.5
         grid_width = (boundary.get_long_max - boundary.get_long_min) / cell_width
         self.neighbour_graph = NeighbourGraph (self.cellboxes ,grid_width )

   def test_initialize_NG(self):
      self.assertEqual ( 4 , len (self.neighbour_graph.get_graph()))
      self.assertEqual ( {1: [], 2: [1], 3: [3], 4: [2], -1: [], -2: [], -3: [], -4: []} , self.neighbour_graph.get_graph()[0]) # NW cellbox
      self.assertEqual ( {1: [], 2: [], 3: [], 4: [3], -1: [2], -2: [0], -3: [], -4: []} , self.neighbour_graph.get_graph()[1]) # NE cellbox
      self.assertEqual ( {1: [1], 2: [3], 3: [], 4: [], -1: [], -2: [], -3: [], -4: [0]} , self.neighbour_graph.get_graph()[2]) # SW cellbox
      self.assertEqual ( {1: [], 2: [], 3: [], 4: [], -1: [], -2: [2], -3: [0], -4: [1]} , self.neighbour_graph.get_graph()[3]) # SE cellbox

   
   def test_remove_neighbour (self):
      self.neighbour_graph.remove_neighbour(0 ,  Direction.south_east , 3)
      self.assertEqual ( {1: [], 2: [1], 3: [], 4: [2], -1: [], -2: [], -3: [], -4: []} , self.neighbour_graph.get_graph()[0]) # NW cellbox

   def test_update_neighbour (self):
      self.neighbour_graph.update_neighbour(0 ,  Direction.south_east , [3])
      self.assertEqual ( {1: [], 2: [1], 3: [3], 4: [2], -1: [], -2: [], -3: [], -4: []} , self.neighbour_graph.get_graph()[0]) # NW cellbox
      
   def test_get_neighbour_case(self):
      case = self.neighbour_graph.get_neighbour_case(self.cellboxes[0] , self.cellboxes[1])
      self.assertEqual ( 2, case)
      case = self.neighbour_graph.get_neighbour_case(self.cellboxes[0] , self.cellboxes[2])
      self.assertEqual ( 4, case)
      case = self.neighbour_graph.get_neighbour_case(self.cellboxes[0] , self.cellboxes[3])
      self.assertEqual (3 , case)
      case = self.neighbour_graph.get_neighbour_case(self.cellboxes[2] , self.cellboxes[0])
      self.assertEqual ( -4, case)
      case = self.neighbour_graph.get_neighbour_case(self.cellboxes[1] , self.cellboxes[2])
      self.assertEqual ( -1, case)
      case = self.neighbour_graph.get_neighbour_case(self.cellboxes[2] , self.cellboxes[1])
      self.assertEqual ( 1, case)
      case = self.neighbour_graph.get_neighbour_case(self.cellboxes[3] , self.cellboxes[0])
      self.assertEqual ( -3, case)
      case = self.neighbour_graph.get_neighbour_case(self.cellboxes[3] , self.cellboxes[2])
      self.assertEqual ( -2, case)
   
   def test_update_neighbours (self):
      self.neighbour_graph.update_neighbours(0, [2,4], Direction.south_east, self.cellboxes)
      self.assertEqual ( {1: [], 2: [], 3: [], 4: [], -1: [], -2: [2], -3: [2,4], -4: [1]} , self.neighbour_graph.get_graph()[3]) # SE cellbox
      self.neighbour_graph.update_neighbour(3, Direction.north_west , [0]) # undo the previous line
   
   #TODO: fill
   #def test_update_corner_neighbours(self):
   #   self.neighbour_graph.update_corner_neighbours (0 , )
