import unittest
from fastapi.testclient import TestClient
from main import app


class TestFastAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/voltForge-ai/api/v1/model/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "UP")

    def test_chat_endpoint(self):
        payload = {
            "message": "Is my ESP32 circuit safe?",
            "boardType": "ESP32",
            "components": [{"type": "MPU6050"}]
        }
        response = self.client.post("/voltForge-ai/api/v1/model/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("reply", data)

    def test_validate_circuit_endpoint(self):
        payload = {
            "boardType": "ARDUINO_UNO",
            "components": [{"type": "LED"}]
        }
        response = self.client.post("/voltForge-ai/api/v1/model/validate-circuit", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("safetyScore", data)

    def test_generate_code_endpoint(self):
        payload = {
            "boardType": "ARDUINO_UNO",
            "components": [{"type": "LED", "pin": "13"}]
        }
        response = self.client.post("/voltForge-ai/api/v1/model/generate-code", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("generatedCode", data)


if __name__ == "__main__":
    unittest.main()
