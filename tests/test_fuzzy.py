import unittest
from engine.fuzzy_resolver import FuzzyResolver


class TestFuzzyResolver(unittest.TestCase):

    def test_board_typo_resolution(self):
        self.assertEqual(FuzzyResolver.resolve_board("arduno"), "ARDUINO_UNO")
        self.assertEqual(FuzzyResolver.resolve_board("esp 32"), "ESP32")
        self.assertEqual(FuzzyResolver.resolve_board("rasbery pi pico"), "RASPBERRY_PI_PICO")
        self.assertEqual(FuzzyResolver.resolve_board("blue pill"), "STM32_BLUE_PILL")

    def test_component_typo_resolution(self):
        self.assertEqual(FuzzyResolver.resolve_component("mpu 6050"), "MPU6050")
        self.assertEqual(FuzzyResolver.resolve_component("oled dispay"), "OLED_DISPLAY")
        self.assertEqual(FuzzyResolver.resolve_component("rely"), "RELAY_MODULE")
        self.assertEqual(FuzzyResolver.resolve_component("ledd"), "LED")

    def test_text_normalization(self):
        res = FuzzyResolver.normalize_text("check my arduno with mpu 6050")
        self.assertIn("ARDUINO_UNO", res["resolvedBoards"])
        self.assertIn("MPU6050", res["resolvedComponents"])


if __name__ == "__main__":
    unittest.main()
