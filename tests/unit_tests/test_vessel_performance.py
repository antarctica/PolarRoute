import unittest
from copy import copy
import numpy as np
from polar_route.vessel_performance.vessels.SDA import SDA, wind_resistance, wind_mag_dir, c_wind, calc_wind, fuel_eq
from cartographi.mesh_generation.aggregated_cellbox import AggregatedCellBox
from cartographi.mesh_generation.boundary import Boundary


class TestSDA(unittest.TestCase):
    def setUp(self):
        config = {
            "VesselType": "SDA",
            "MaxSpeed": 26.5,
            "Unit": "km/hr",
            "Beam": 24.0,
            "HullType": "slender",
            "ForceLimit": 96634.5,
            "MaxIceConc": 80,
            "MinDepth": -10
            }
        boundary = Boundary([-85, -84.9], [-135, -134.9], ['1970-01-01', '2021-12-31'])

        self.cellbox = AggregatedCellBox(boundary, {}, '0')
        self.SDA = SDA(config)

    def test_land(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'elevation': -100.
        }

        actual = self.SDA.land(cellbox)
        expected = False
        self.assertEqual(actual, expected)

    def test_extreme_ice(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'SIC': 100.
        }

        actual = self.SDA.extreme_ice(cellbox)
        expected = True
        self.assertEqual(actual, expected)

    def test_model_speed_open_water(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': 26.5,
            'SIC': 0.,
            'thickness': 0.,
            'density': 0.
        }

        actual = self.SDA.model_speed(cellbox).agg_data

        expected = {
            'speed': [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5],
            'SIC': 0.,
            'thickness': 0.,
            'density': 0.,
            'ice resistance': 0.
        }

        self.assertEqual(actual, expected)

    def test_model_speed_ice(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': 26.5,
            'SIC': 60.,
            'thickness': 1.,
            'density': 980.
        }

        actual = self.SDA.model_speed(cellbox).agg_data

        expected = {
            'speed': [7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933,
                      7.842665122593933, 7.842665122593933, 7.842665122593933],
            'SIC': 60.,
            'thickness': 1.,
            'density': 980.,
            'ice resistance': 96634.5,
        }

        self.assertEqual(actual, expected)

    def test_model_resistance_open_water(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5],
            'SIC': 0.,
            'thickness': 0.,
            'density': 0.,
            'ice resistance': 0.
        }

        actual =self.SDA.model_resistance(cellbox).agg_data

        expected = {
            'speed': [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5],
            'SIC': 0.,
            'thickness': 0.,
            'density': 0.,
            'ice resistance': 0.,
            'resistance': [0., 0., 0., 0., 0., 0., 0., 0.]
        }

        self.assertEqual(actual, expected)

    def test_model_resistance_ice(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933,
                      7.842665122593933, 7.842665122593933, 7.842665122593933],
            'SIC': 60.,
            'thickness': 1.,
            'density': 980.,
            'ice resistance': 96634.5,
        }

        actual =self.SDA.model_resistance(cellbox).agg_data

        expected = {
            'speed': [7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933,
                      7.842665122593933, 7.842665122593933, 7.842665122593933],
            'SIC': 60.,
            'thickness': 1.,
            'density': 980.,
            'ice resistance': 96634.5,
            'resistance': [96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5]
        }

        self.assertEqual(actual, expected)

    def test_model_fuel_open_water(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5],
            'SIC': 0.,
            'thickness': 0.,
            'density': 0.,
            'ice resistance': 0.,
            'resistance': [0., 0., 0., 0., 0., 0., 0., 0.]
        }

        actual =self.SDA.model_fuel(cellbox).agg_data

        expected = {
            'speed': [26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5],
            'SIC': 0.,
            'thickness': 0.,
            'density': 0.,
            'ice resistance': 0.,
            'resistance': [0., 0., 0., 0., 0., 0., 0., 0.],
            'fuel': [27.3186897, 27.3186897, 27.3186897, 27.3186897, 27.3186897, 27.3186897, 27.3186897, 27.3186897]
        }

        self.assertEqual(actual, expected)

    def test_model_fuel_ice(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933,
                      7.842665122593933, 7.842665122593933, 7.842665122593933],
            'SIC': 60.,
            'thickness': 1.,
            'density': 980.,
            'ice resistance': 96634.5,
            'resistance': [96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5]
        }

        actual =self.SDA.model_fuel(cellbox).agg_data

        expected = {
            'speed': [7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933,
                      7.842665122593933, 7.842665122593933, 7.842665122593933],
            'SIC': 60.0,
            'thickness': 1.0,
            'density': 980.0,
            'ice resistance': 96634.5,
            'resistance': [96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5, 96634.5],
            'fuel': [39.94376930737089, 39.94376930737089, 39.94376930737089, 39.94376930737089, 39.94376930737089,
                     39.94376930737089, 39.94376930737089, 39.94376930737089]
        }


        self.assertEqual(actual, expected)

    def test_ice_resistance_zero(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': 0.,
            'SIC': 0.,
            'thickness': 0.,
            'density':0.
        }

        actual = self.SDA.ice_resistance(cellbox)
        expected = 0.
        self.assertEqual(actual, expected)

    def test_ice_resistance_pos(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': 5.56,
            'SIC': 60.,
            'thickness': 1.,
            'density': 980.
        }

        actual = self.SDA.ice_resistance(cellbox)
        expected = 64543.75549708632
        self.assertAlmostEqual(actual, expected, places=5)

    def test_invert_resistance_zero(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': 26.5,
            'SIC': 0.,
            'thickness': 0.,
            'density': 0.
        }

        actual = self.SDA.invert_resistance(cellbox)
        expected = 26.5
        self.assertAlmostEqual(actual, expected, places=5)

    def test_invert_resistance_pos(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': 26.5,
            'SIC': 60.,
            'thickness': 1.,
            'density': 980.
        }

        actual = self.SDA.invert_resistance(cellbox)
        expected = 7.842665122593933
        self.assertAlmostEqual(actual, expected, places=5)

    def test_fuel_eq_hotel(self):
        actual = fuel_eq(0., 0.)
        expected = 6.06970392
        self.assertAlmostEqual(actual, expected, places=5)

    def test_fuel_eq_open_water(self):
        actual = fuel_eq(26.5, 0)
        expected = 27.3186897
        self.assertAlmostEqual(actual, expected, places=5)

    def test_fuel_eq_ice_breaking(self):
        actual = fuel_eq(5.56, 64543.76)
        expected = 24.48333122037351
        self.assertAlmostEqual(actual, expected, places=5)

    def test_wind_coeff_zero(self):
        actual = c_wind(0.)
        expected = 0.94
        self.assertEqual(actual, expected)

    def test_wind_coeff_interp(self):
        actual = c_wind(np.pi/4.)
        expected = 0.595
        self.assertEqual(actual, expected)

    def test_wind_resistance_zero(self):
        actual = wind_resistance(0.,0.,0.)
        expected = 0.
        self.assertEqual(actual, expected)

    def test_wind_resistance_equal_opposite(self):
        actual = wind_resistance(10., 10., 0.)
        expected = 0.
        self.assertEqual(actual, expected)

    def test_wind_resistance_pos(self):
        actual = wind_resistance(10., 20., 0.)
        expected = 129543.75
        self.assertAlmostEqual(actual, expected, places=5)

    def test_wind_resistance_neg(self):
        actual = wind_resistance(10., 20., np.pi)
        expected = -105656.25
        self.assertAlmostEqual(actual, expected, places=5)

    def test_wind_mag_dir_zero(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [0., 0., 0., 0., 0., 0., 0., 0.],
            'u10': 0.,
            'v10': 0.
        }
        actual = wind_mag_dir(cellbox, 0.)
        expected = (0., 0.)
        self.assertEqual(actual, expected)

    def test_wind_mag_dir_north(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [10., 10., 10., 10., 10., 10., 10., 10.],
            'u10': 0.,
            'v10': 10.
        }
        actual = wind_mag_dir(cellbox, 0.)
        expected = (7.22222, np.pi)
        self.assertAlmostEqual(actual[0], expected[0], places=5)
        self.assertAlmostEqual(actual[1], expected[1], places=5)

    def test_wind_mag_dir_east(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [10., 10., 10., 10., 10., 10., 10., 10.],
            'u10': 10.,
            'v10': 0.
        }
        actual = wind_mag_dir(cellbox, 0.)
        expected = (10.378634, 1.299849)
        self.assertAlmostEqual(actual[0], expected[0], places=5)
        self.assertAlmostEqual(actual[1], expected[1], places=5)

    def test_calc_wind_zero(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [0., 0., 0., 0., 0., 0., 0., 0.],
            'u10': 0.,
            'v10': 0.
        }
        actual = calc_wind(cellbox).agg_data

        expected = {
            'speed': [0., 0., 0., 0., 0., 0., 0., 0.],
            'u10': 0.,
            'v10': 0.,
            'wind resistance': [0, 0, 0, 0, 0, 0, 0, 0],
            'relative wind speed': [0, 0, 0, 0, 0, 0, 0, 0],
            'relative wind angle': [0, 0, 0, 0, 0, 0, 0, 0]
        }
        self.assertEqual(actual, expected)

    def test_calc_wind_north(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [10., 10., 10., 10., 10., 10., 10., 10.],
            'u10': 0.,
            'v10': 10.
        }
        actual = calc_wind(cellbox).agg_data

        expected = {
            'speed': [10., 10., 10., 10., 10., 10., 10., 10.],
            'u10': 0.0,
            'v10': 10.0,
            'wind resistance': [1389.4514464984172, 18883.168370819058, 44192.363791332944, 67170.852525, 44192.36379133295, 18883.168370819058, 1389.45144649843, -11478.704021299203],
            'relative wind speed': [8.272382984093076, 10.378634868247365, 12.124347537962088, 12.77778, 12.124347537962088, 10.378634868247365, 8.272382984093076, 7.22222],
            'relative wind angle': [2.11646579563727, 1.299849270152763, 0.6226774988830187, 0.0, 0.6226774988830185, 1.2998492701527629, 2.1164657956372697, 3.141592653589793]
        }

        self.assertEqual(actual, expected)

    def test_calc_wind_east(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [10., 10., 10., 10., 10., 10., 10., 10.],
            'u10': 10.,
            'v10': 0.
        }
        actual = calc_wind(cellbox).agg_data

        expected = {
            'speed': [10., 10., 10., 10., 10., 10., 10., 10.],
            'u10': 10.0,
            'v10': 0.0,
            'wind resistance': [1389.45144649843, -11478.704021299203, 1389.4514464984172, 18883.168370819058, 44192.363791332944, 67170.852525, 44192.363791332944, 18883.168370819058],
            'relative wind speed': [8.272382984093076, 7.22222, 8.272382984093076, 10.378634868247365, 12.124347537962088, 12.77778, 12.124347537962088, 10.378634868247365],
            'relative wind angle': [2.1164657956372697, 3.141592653589793, 2.11646579563727, 1.299849270152763, 0.6226774988830187, 0.0, 0.6226774988830187, 1.2998492701527629]
        }

        self.assertEqual(actual, expected)

    def test_model_resistance_ice_wind_north(self):
        cellbox = copy(self.cellbox)
        cellbox.agg_data = {
            'speed': [7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933,
                      7.842665122593933, 7.842665122593933, 7.842665122593933],
            'SIC': 60.,
            'thickness': 1.,
            'density': 980.,
            'ice resistance': 96634.5,
            'u10': 0.,
            'v10': 10.
        }

        actual = self.SDA.model_resistance(cellbox).agg_data

        expected = {
            'speed': [7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933, 7.842665122593933,
                      7.842665122593933, 7.842665122593933, 7.842665122593933],
            'SIC': 60.0,
            'thickness': 1.0,
            'density': 980.0,
            'ice resistance': 96634.5,
            'u10': 0.0,
            'v10': 10.0,
            'wind resistance': [1234.4775504276085, 19864.391781102568, 40525.12205230855, 61995.4919027709, 40525.1220523086, 19864.391781102568, 1234.4775504276222, -11604.216485701227],
            'relative wind speed': [8.598664182949458, 10.234546822417895, 11.64280342483676, 12.178519832423898, 11.642803424836762, 10.234546822417895, 8.598664182949458, 7.8214801675761025],
            'relative wind angle': [2.176072621553403, 1.356295795185576, 0.652700196811307, 0.0, 0.6527001968113064, 1.3562957951855759, 2.1760726215534025, 3.141592653589793],
            'resistance': [97868.9775504276, 116498.89178110257, 137159.62205230855, 158629.99190277088, 137159.6220523086, 116498.89178110257, 97868.97755042762, 85030.28351429878]
        }

        self.assertEqual(actual, expected)
