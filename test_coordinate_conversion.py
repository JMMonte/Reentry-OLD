import unittest
import numpy as np
from coordinate_converter import (
    gst_to_rotation_matrix, ecef_to_enu_rotation_matrix, gst_to_angular_velocity_matrix, inverse_3x3_matrix,
    eci_to_ecef, ecef_to_eci, geodetic_to_spheroid, geodetic_to_ecef, ecef_to_geodetic,
    eci_velocity_to_ground_velocity, ecef_to_enu, ecef_distance
)

class TestCoordinateConversion(unittest.TestCase):
    def test_gst_to_rotation_matrix(self):
        gst = 1.0
        expected = np.array([[0.5403023059, 0.8414709848, 0],
                             [-0.84147098, 0.54030231, 0],
                             [0, 0, 1]])
        result = gst_to_rotation_matrix(gst)
        np.testing.assert_array_almost_equal(expected, result)

    def test_ecef_to_enu_rotation_matrix(self):
        lat, lon = 40.7128, -74.0060
        expected = np.array([[-0.0359, 0.9993, 0],
                             [0.7393, 0.0251, 0.6730],
                             [-0.6724, -0.0291, 0.7396]])
        result = ecef_to_enu_rotation_matrix(lat, lon)
        np.testing.assert_array_almost_equal(expected, result, decimal=4)

    def test_gst_to_angular_velocity_matrix(self):
        gst = 1.0
        expected = np.array([[-np.sin(gst), np.cos(gst), 0],
                             [-np.cos(gst), -np.sin(gst), 0],
                             [0, 0, 0]])
        result = gst_to_angular_velocity_matrix(gst)
        np.testing.assert_array_almost_equal(expected, result)

    def test_inverse_3x3_matrix(self):
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        expected = np.array([[-2, 1, 0], [1.5, -0.5, -0.5], [0.5, 0, 0]])
        result = inverse_3x3_matrix(mat)
        np.testing.assert_array_almost_equal(expected, result)

    def test_eci_to_ecef(self):
        r_eci = np.array([1, 2, 3])
        v_eci = np.array([4, 5, 6])
        gst = 1.0
        r_ecef_expected = np.array([1.0806, 3.4532, 3.0])
        v_ecef_expected = np.array([1.4241, 7.9203, 6.0])
        r_ecef_result, v_ecef_result = eci_to_ecef(r_eci, v_eci, gst)
        np.testing.assert_array_almost_equal(r_ecef_expected, r_ecef_result, decimal=4)
        np.testing.assert_array_almost_equal(v_ecef_expected, v_ecef_result, decimal=4)

    def test_ecef_to_eci(self):
        r_ecef = np.array([1, 2, 3])
        v_ecef = np.array([4, 5, 6])
        gst = 1.0
        r_eci_expected = np.array([0.9194, 0.5468, 3.0])
        v_eci_expected = np.array([3.5759, 2.0797, 6.0])
        r_eci_result, v_eci_result = ecef_to_eci(r_ecef, v_ecef, gst)
        np.testing.assert_array_almost_equal(r_eci_expected, r_eci_result, decimal=4)
        np.testing.assert_array_almost_equal(v_eci_expected, v_eci_result, decimal=4)

    def test_geodetic_to_spheroid(self):
        lat, lon, alt = 0.5, 0.5, 1
        x, y, z = 6378138.8515, 6378138.8515, 1.6375
        result = geodetic_to_spheroid(lat, lon, alt)
        np.testing.assert_array_almost_equal((x, y, z), result, decimal=4)

    def test_geodetic_to_ecef(self):
        lat, lon, alt = 40.7128, -74.0060, 100
        x, y, z = -1333981.9011, -4654180.6006, 4136.6610
        result = geodetic_to_ecef(lat, lon, alt)
        np.testing.assert_array_almost_equal((x, y, z), result, decimal=4)

    def test_ecef_to_geodetic(self):
        x, y, z = -1333981.9011, -4654180.6006, 4136.6610
        lat, lon, alt = 40.7128, -74.0060, 100
        result = ecef_to_geodetic(x, y, z)
        np.testing.assert_array_almost_equal((lat, lon, alt), result, decimal=4)

    def test_eci_velocity_to_ground_velocity(self):
        v_eci = np.array([4, 5, 6])
        lat, lon = 40.7128, -74.0060
        gst = 1.0
        v_ground_expected = np.array([1.4241, 7.9203, 6.0])
        v_ground_result = eci_velocity_to_ground_velocity(v_eci, lat, lon, gst)
        np.testing.assert_array_almost_equal(v_ground_expected, v_ground_result, decimal=4)

    def test_ecef_to_enu(self):
        x, y, z = -1333981.9011, -4654180.6006, 4136.6610
        lat, lon = 40.7128, -74.0060
        e, n, u = 0.0, 0.0, 0.0
        result = ecef_to_enu(x, y, z, lat, lon)
        np.testing.assert_array_almost_equal((e, n, u), result, decimal=4)

    def test_ecef_distance(self):
        x1, y1, z1 = -1333981.9011, -4654180.6006, 4136.6610
        x2, y2, z2 = -1333981.9011, -4654180.6006, 4136.6610
        expected = 0.0
        result = ecef_distance(x1, y1, z1, x2, y2, z2)
        np.testing.assert_array_almost_equal(expected, result, decimal=4)


if __name__ == "__main__":
    unittest.main()
