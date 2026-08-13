import unittest
from engine.physics_solver import ElectricalPhysicsSolver


class TestElectricalPhysicsSolver(unittest.TestCase):

    def test_ohms_law(self):
        res = ElectricalPhysicsSolver.calculate_ohms_law(voltage=5.0, current=0.02)
        self.assertEqual(res["resistance"], 250.0)
        self.assertEqual(res["power"], 0.1)

    def test_led_resistor(self):
        res = ElectricalPhysicsSolver.calculate_led_resistor(5.0, 2.0, 20.0)
        self.assertEqual(res["exactResistance"], 150.0)
        self.assertEqual(res["recommendedResistor"], "150 ohm")
        self.assertTrue(res["isSafeQuarterWatt"])

    def test_voltage_divider(self):
        res = ElectricalPhysicsSolver.calculate_voltage_divider(5.0, 10000.0, 20000.0)
        self.assertEqual(res["vOut"], 3.333)
        self.assertAlmostEqual(res["currentDrawmA"], 0.167, places=2)

    def test_rc_cutoff(self):
        res = ElectricalPhysicsSolver.calculate_rc_cutoff(10000.0, 1e-7)
        self.assertAlmostEqual(res["cutoffFrequencyHz"], 159.15, places=1)


if __name__ == "__main__":
    unittest.main()
