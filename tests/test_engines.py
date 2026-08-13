import unittest
from engine.firmware_analyzer import FirmwareAnalyzer


class TestFirmwareAnalyzer(unittest.TestCase):

    def test_pin_collision(self):
        code = "#define LED_PIN 13\n#define SENSOR_PIN 13\nvoid setup() {}\nvoid loop() {}"
        res = FirmwareAnalyzer.analyze(code)
        self.assertTrue(any(i["type"] == "PIN_COLLISION" for i in res["issues"]))

    def test_missing_include(self):
        code = "#include <Arduino.h>\nvoid setup() {}\nvoid loop() {}"
        res = FirmwareAnalyzer.analyze(code, [{"type": "MPU6050"}])
        self.assertTrue(any(i["type"] == "MISSING_INCLUDE" for i in res["issues"]))

    def test_blocking_delay(self):
        code = "void setup() {}\nvoid loop() { delay(2000); }"
        res = FirmwareAnalyzer.analyze(code)
        self.assertTrue(any(i["type"] == "BLOCKING_DELAY" for i in res["issues"]))

    def test_missing_pinmode(self):
        code = "void setup() {}\nvoid loop() { digitalWrite(LED_PIN, HIGH); }"
        res = FirmwareAnalyzer.analyze(code)
        self.assertTrue(any(i["type"] == "MISSING_PINMODE" for i in res["issues"]))

    def test_clean_code_high_score(self):
        code = "#include <Arduino.h>\n#define LED_PIN 13\nvoid setup() { pinMode(LED_PIN, OUTPUT); }\nvoid loop() { delay(100); }"
        res = FirmwareAnalyzer.analyze(code)
        self.assertGreaterEqual(res["score"], 80)


class TestKnowledgeGraph(unittest.TestCase):
    def test_get_mpu6050(self):
        from engine.knowledge_graph import ComponentKnowledgeGraph
        info = ComponentKnowledgeGraph.get_component_info("MPU6050")
        self.assertIsNotNone(info)
        self.assertIn("0x68", info["i2cAddress"])

    def test_get_equivalents(self):
        from engine.knowledge_graph import ComponentKnowledgeGraph
        eq = ComponentKnowledgeGraph.get_equivalents("DHT22")
        self.assertIn("SHT30", eq)


class TestTroubleshooter(unittest.TestCase):
    def test_i2c_diagnosis(self):
        from engine.troubleshooter import TroubleshootingEngine
        res = TroubleshootingEngine.diagnose("MPU6050 not detected", "ESP32", [{"type": "MPU6050"}])
        self.assertEqual(res["diagnosis"][0]["category"], "I2C_BUS_FAILURE")

    def test_led_diagnosis(self):
        from engine.troubleshooter import TroubleshootingEngine
        res = TroubleshootingEngine.diagnose("LED not lighting up", "ARDUINO_UNO", [{"type": "LED"}])
        self.assertEqual(res["diagnosis"][0]["category"], "LED_NOT_WORKING")


class TestNLSynthesizer(unittest.TestCase):
    def test_weather_station(self):
        from engine.nl_synthesizer import NLCircuitSynthesizer
        res = NLCircuitSynthesizer.synthesize("Build a weather station with OLED display", "ESP32")
        self.assertTrue(res["success"])
        types = {c["type"] for c in res["components"]}
        self.assertIn("BME280", types)
        self.assertIn("OLED_DISPLAY", types)

    def test_bom_export(self):
        from engine.nl_synthesizer import BOMExporter
        bom = BOMExporter.generate_bom([{"type": "LED"}, {"type": "RESISTOR"}])
        self.assertGreater(bom["totalCost_USD"], 0)


class TestBusSolvers(unittest.TestCase):
    def test_can_termination(self):
        from engine.bus_solver import CANBusSolver
        res = CANBusSolver.calculate_termination(5.0, 3)
        self.assertTrue(res["needsTermination"])

    def test_i2c_collision(self):
        from engine.bus_solver import I2CMultiplexerSolver
        devices = [{"name": "BME280_1", "address": "0x76"}, {"name": "BME280_2", "address": "0x76"}]
        res = I2CMultiplexerSolver.resolve_collisions(devices)
        self.assertEqual(res["collisionCount"], 1)

    def test_modbus_crc(self):
        from engine.bus_solver import ModbusCRCSolver
        res = ModbusCRCSolver.calculate_crc16([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        self.assertIn("crc16", res)


class TestPowerManager(unittest.TestCase):
    def test_battery_life(self):
        from engine.power_manager import BatteryLifeEstimator
        res = BatteryLifeEstimator.estimate_runtime(2000, 80, 10, 5)
        self.assertGreater(res["estimatedRuntime_hours"], 0)

    def test_ldo_thermal(self):
        from engine.power_manager import LDOThermalSolver
        res = LDOThermalSolver.calculate(12.0, 3.3, 200)
        self.assertIn("isSafe", res)

    def test_solar_sizing(self):
        from engine.power_manager import SolarPanelSizer
        res = SolarPanelSizer.size_panel(500, 3.7, 4.0)
        self.assertGreater(res["recommendedPanelPower_W"], 0)


class TestWirelessSolvers(unittest.TestCase):
    def test_wifi_power(self):
        from engine.wireless_solver import WiFiPowerOptimizer
        res = WiFiPowerOptimizer.estimate_power("deep_sleep")
        self.assertEqual(res["current_mA"], 0.01)

    def test_fspl(self):
        from engine.wireless_solver import FreeSpacePathLossCalculator
        res = FreeSpacePathLossCalculator.calculate_fspl(100, 2400)
        self.assertGreater(res["freeSpacePathLoss_dB"], 50)

    def test_rssi(self):
        from engine.wireless_solver import RSSIInterpreter
        res = RSSIInterpreter.interpret(-55)
        self.assertEqual(res["signalQuality"], "Good")


if __name__ == "__main__":
    unittest.main()
