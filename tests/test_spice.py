import unittest
from engine.spice_engine import (
    SpiceNetlistExporter, DCOperatingPointSolver, TransientAnalysisSolver,
    ACFrequencyResponseSolver, Timer555Solver, PWMDutyCycleCalculator
)


class TestSpiceNetlistExporter(unittest.TestCase):
    def test_basic_netlist(self):
        netlist = SpiceNetlistExporter.export_netlist(
            [{"type": "RESISTOR", "value": "1k"}, {"type": "LED"}], [], "ARDUINO_UNO", 5.0
        )
        self.assertIn("Voltforge SPICE Netlist", netlist)
        self.assertIn("R1", netlist)
        self.assertIn("D2", netlist)
        self.assertIn(".op", netlist)


class TestDCOperatingPoint(unittest.TestCase):
    def test_series_resistors(self):
        res = DCOperatingPointSolver.solve_series_resistors(5.0, [220, 330])
        self.assertEqual(res["totalResistance"], 550)
        self.assertAlmostEqual(res["totalCurrent_mA"], 9.0909, places=2)

    def test_parallel_resistors(self):
        res = DCOperatingPointSolver.solve_parallel_resistors(5.0, [1000, 1000])
        self.assertEqual(res["equivalentResistance"], 500.0)


class TestTransientAnalysis(unittest.TestCase):
    def test_rc_charging(self):
        res = TransientAnalysisSolver.rc_charging(5.0, 10000, 100e-6, 5.0)
        self.assertAlmostEqual(res["capacitorVoltage_V"], 5.0, places=1)

    def test_rl_step(self):
        res = TransientAnalysisSolver.rl_step_response(5.0, 100, 0.01, 1.0)
        self.assertGreater(res["inductorCurrent_mA"], 0)


class TestTimer555(unittest.TestCase):
    def test_astable(self):
        res = Timer555Solver.astable(10000, 47000, 10e-6)
        self.assertGreater(res["frequency_Hz"], 0)
        self.assertGreater(res["dutyCyclePercent"], 50)


class TestPWM(unittest.TestCase):
    def test_duty_cycle(self):
        res = PWMDutyCycleCalculator.calculate(5.0, 50.0)
        self.assertEqual(res["vAverage"], 2.5)


if __name__ == "__main__":
    unittest.main()
