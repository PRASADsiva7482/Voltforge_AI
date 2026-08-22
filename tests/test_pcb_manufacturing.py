import io
import unittest
import zipfile

from engine.drc_engine import PcbDrcEngine
from engine.gerber_exporter import GerberExporter


class TestPcbManufacturingEngines(unittest.TestCase):

    def test_drc_flags_trace_width_and_clearance(self):
        result = PcbDrcEngine.run_drc(
            board_width_mm=40,
            board_height_mm=30,
            footprints=[],
            traces=[
                {
                    "id": "t1",
                    "netId": "N1",
                    "layer": "F.Cu",
                    "width_mm": 0.10,
                    "points": [{"x": 5, "y": 5}, {"x": 25, "y": 5}],
                },
                {
                    "id": "t2",
                    "netId": "N2",
                    "layer": "F.Cu",
                    "width_mm": 0.254,
                    "points": [{"x": 10, "y": 5.15}, {"x": 30, "y": 5.15}],
                },
            ],
            vias=[],
            wires=[],
        )

        self.assertFalse(result["passed"])
        rules = {violation["rule"] for violation in result["violations"]}
        self.assertIn("Minimum Trace Width", rules)
        self.assertIn("Minimum Copper Clearance", rules)

    def test_gerber_export_contains_required_fabrication_layers(self):
        zip_bytes = GerberExporter.generate_gerber_zip(
            board_width_mm=50,
            board_height_mm=40,
            project_name="Demo Board",
            footprints=[
                {
                    "id": "fp_u1",
                    "name": "U1",
                    "x": 20,
                    "y": 18,
                    "width": 10,
                    "height": 8,
                    "pads": [
                        {"id": "1", "name": "1", "x": -2.54, "y": 2, "width": 1.4, "height": 1.4, "shape": "rect", "drillDiameter": 0.8},
                        {"id": "2", "name": "2", "x": 2.54, "y": 2, "width": 1.4, "height": 1.4, "shape": "circle", "drillDiameter": 0.8},
                    ],
                }
            ],
            traces=[
                {
                    "id": "t1",
                    "netId": "N1",
                    "layer": "F.Cu",
                    "width_mm": 0.254,
                    "points": [{"x": 17.46, "y": 20}, {"x": 22.54, "y": 20}],
                }
            ],
            vias=[{"id": "v1", "x": 30, "y": 22, "drill_mm": 0.3, "pad_mm": 0.6}],
        )

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            names = set(archive.namelist())
            self.assertIn("Demo_Board-F_Cu.gtl", names)
            self.assertIn("Demo_Board-B_Cu.gbl", names)
            self.assertIn("Demo_Board-F_Mask.gts", names)
            self.assertIn("Demo_Board-B_Mask.gbs", names)
            self.assertIn("Demo_Board-F_SilkS.gto", names)
            self.assertIn("Demo_Board-B_SilkS.gbo", names)
            self.assertIn("Demo_Board-Edge_Cuts.gml", names)
            self.assertIn("Demo_Board-PTH.drl", names)
            self.assertIn("manifest.json", names)
            self.assertIn("%MOMM*%", archive.read("Demo_Board-F_Cu.gtl").decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
