import unittest
from circuit_verifier import ElectricalVerifier


class TestElectricalVerifier(unittest.TestCase):

    def test_3v3_voltage_mismatch(self):
        res = ElectricalVerifier.verify_circuit(
            board_type="ESP32",
            components=[{"type": "5V_SENSOR"}]
        )
        self.assertLess(res["safetyScore"], 100)
        self.assertTrue(any(i["severity"] == "CRITICAL" for i in res["issues"]))

    def test_i2c_pullup_check(self):
        res = ElectricalVerifier.verify_circuit(
            board_type="ARDUINO_UNO",
            components=[{"type": "MPU6050"}]
        )
        self.assertTrue(any("I2C" in i["message"] for i in res["issues"]))
        self.assertTrue(any(a["type"] == "RESISTOR" for a in res["additions"]))

    def test_inductive_flyback_check(self):
        res = ElectricalVerifier.verify_circuit(
            board_type="ARDUINO_UNO",
            components=[{"type": "RELAY_MODULE"}]
        )
        self.assertTrue(any("Inductive" in i["message"] for i in res["issues"]))
        self.assertTrue(any("DIODE" in a["type"] for a in res["additions"]))


if __name__ == "__main__":
    unittest.main()
