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

    def test_export_uses_wire_nets(self):
        components = [
            {"id": "src", "type": "DC_SOURCE_5V", "pins": [{"id": "positive", "type": "power"}, {"id": "negative", "type": "ground"}]},
            {"id": "r", "type": "RESISTOR", "resistance": "1k", "pins": [{"id": "p1"}, {"id": "p2"}]},
            {"id": "gnd", "type": "GROUND", "pins": [{"id": "gnd", "type": "ground"}]},
        ]
        wires = [
            {"fromNodeId": "src", "fromPinId": "positive", "toNodeId": "r", "toPinId": "p1"},
            {"fromNodeId": "r", "fromPinId": "p2", "toNodeId": "gnd", "toPinId": "gnd"},
            {"fromNodeId": "src", "fromPinId": "negative", "toNodeId": "gnd", "toPinId": "gnd"},
        ]
        netlist = SpiceNetlistExporter.export_netlist(components, wires)
        self.assertIn("V1 N1 0 DC 5", netlist)
        self.assertIn("R2 N1 0 1k", netlist)

    def test_export_flattens_subcircuit_and_transformer(self):
        components = [
            {"id": "src", "type": "DC_SOURCE_5V", "pins": [{"id": "positive", "type": "power"}, {"id": "negative", "type": "ground"}]},
            {
                "id": "module",
                "type": "SUB_CIRCUIT",
                "pins": [{"id": "subpin_1"}, {"id": "subpin_2", "type": "ground"}],
                "properties": {
                    "subCircuitDef": {
                        "internalNodes": [{"id": "inner_r", "type": "RESISTOR", "resistance": "1k", "pins": [{"id": "p1"}, {"id": "p2"}]}],
                        "internalWires": [{"fromNodeId": "inner_r", "fromPinId": "p2", "toNodeId": "inner_r", "toPinId": "p2"}],
                        "exposedConnections": {
                            "subpin_1": {"nodeId": "inner_r", "pinId": "p1"},
                            "subpin_2": {"nodeId": "inner_r", "pinId": "p2"},
                        },
                    }
                },
            },
            {"id": "gnd", "type": "GROUND", "pins": [{"id": "gnd", "type": "ground"}]},
            {"id": "xf", "type": "TRANSFORMER", "turnsRatio": 0.5, "pins": [{"id": "primary1"}, {"id": "primary2"}, {"id": "secondary1"}, {"id": "secondary2"}]},
        ]
        wires = [
            {"fromNodeId": "src", "fromPinId": "positive", "toNodeId": "module", "toPinId": "subpin_1"},
            {"fromNodeId": "module", "fromPinId": "subpin_2", "toNodeId": "gnd", "toPinId": "gnd"},
            {"fromNodeId": "src", "fromPinId": "negative", "toNodeId": "gnd", "toPinId": "gnd"},
        ]
        netlist = SpiceNetlistExporter.export_netlist(components, wires)
        self.assertRegex(netlist, r"R\d+ N1 0 1k")
        self.assertRegex(netlist, r"L\d+P")
        self.assertRegex(netlist, r"L\d+S")
        self.assertRegex(netlist, r"K\d+")


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
