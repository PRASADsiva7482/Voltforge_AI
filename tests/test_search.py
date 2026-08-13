import unittest
from web_search_engine import ComponentSpecExtractor, WebSearchEngine


class TestWebSearchEngine(unittest.TestCase):

    def test_spec_extractor(self):
        snippets = [
            {"title": "MPU6050 Datasheet", "snippet": "Operating voltage range 2.3V to 3.4V. Features I2C address 0x68 or 0x69."}
        ]
        specs = ComponentSpecExtractor.extract_specs("MPU6050", snippets)
        self.assertIn("I2C", specs["supportedInterfaces"])
        self.assertIn("0x68", specs["i2cAddresses"])

    def test_web_search_cached_lookup(self):
        engine = WebSearchEngine()
        info = engine.get_component_info("MPU6050")
        self.assertIn("component", info)
        self.assertEqual(info["component"], "MPU6050")


if __name__ == "__main__":
    unittest.main()
