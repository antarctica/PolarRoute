import unittest
import numpy as np
from polar_route.route_planner.crossing import traveltime_in_cell


class TestRouteCalc(unittest.TestCase):

    def test_traveltime_in_cell(self):
        bx = 30
        by = 15
        cx = 2
        cy = -4
        s = np.sqrt(65)

        tt = traveltime_in_cell(bx, by, cx, cy, s)

        expected_tt = 5.0

        self.assertAlmostEqual(tt, expected_tt, places=5)

    def test_traveltime_in_cell_reverse_current(self):
        bx = 30
        by = 15
        cx = -2
        cy = -4
        s = np.sqrt(65)

        tt = traveltime_in_cell(bx, by, cx, cy, s)

        expected_tt = 8.33333

        self.assertAlmostEqual(tt, expected_tt, places=5)

    def test_traveltime_in_cell_slow_vessel(self):
        bx = 30
        by = 15
        cx = 2
        cy = -4
        s = 5

        tt = traveltime_in_cell(bx, by, cx, cy, s)

        expected_tt = 15.0

        self.assertAlmostEqual(tt, expected_tt, places=5)
