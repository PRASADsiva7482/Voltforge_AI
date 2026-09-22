import unittest
import uuid
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

    def test_hardware_coverage_endpoint(self):
        response = self.client.get("/voltForge-ai/api/v1/model/hardware-coverage")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["reportId"], "vfai-fu-001-ui-hardware-coverage")
        self.assertEqual(data["entryCount"], 49)
        self.assertEqual(data["summary"]["verified"], 8)
        entries = {entry["boardType"]: entry for entry in data["entries"]}
        self.assertEqual(entries["ARDUINO_NANO_EVERY"]["status"], "verified")
        self.assertEqual(entries["ARDUINO_NANO"]["status"], "variant-required")

    def test_component_coverage_endpoint(self):
        response = self.client.get("/voltForge-ai/api/v1/model/component-coverage")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["reportId"], "vfai-fu-011-ui-component-coverage")
        self.assertEqual(data["entryCount"], 60)
        self.assertEqual(data["summary"]["verified"], 0)
        self.assertEqual(data["summary"]["variantRequired"], 51)
        entries = {entry["componentType"]: entry for entry in data["entries"]}
        self.assertEqual(entries["OLED_DISPLAY"]["status"], "variant-required")
        self.assertEqual(entries["OSCILLOSCOPE"]["status"], "simulation-only")

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


    def test_system_health_endpoint(self):
        response = self.client.get("/voltForge-ai/api/v1/model/system/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("features", data)

    def test_datasheet_search_endpoint(self):
        payload = {"query": "ESP32", "limit": 2}
        response = self.client.post("/voltForge-ai/api/v1/model/datasheet/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "ESP32")

    def test_feedback_endpoint(self):
        payload = {
            "requestId": f"request-feedback-api-{uuid.uuid4().hex}",
            "responseRecordId": "vf-task-v1-0123456789abcdef01234567",
            "projectId": "project-feedback-api",
            "projectRevision": "client:feedback-api-revision",
            "artifactId": "vfdlm-g1-edge-v1.0.0-test",
            "feedbackKind": "useful",
            "rating": 5,
            "evidence": "The response explained the circuit boundary clearly.",
            "evidenceApproved": True,
        }
        response = self.client.post(
            "/voltForge-ai/api/v1/model/feedback",
            json=payload,
            headers={
                "X-Voltforge-User-Id": "user-feedback-api",
                "X-Voltforge-Project-Id": "project-feedback-api",
            },
        )
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertEqual(data["status"], "PENDING_REVIEW")
        self.assertEqual(data["rawContentStored"], False)

    def test_simulation_stream_endpoint(self):
        payload = {
            "boardType": "ARDUINO_UNO",
            "probes": ["VCC", "D13"],
            "durationMs": 100,
            "sampleRateHz": 50
        }
        response = self.client.post("/voltForge-ai/api/v1/model/simulation/stream", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

    def test_export_circuit_endpoint(self):
        payload = {
            "boardType": "ARDUINO_UNO",
            "components": [
                {"type": "RESISTOR", "value": "220"},
                {"type": "LED", "value": "RED"}
            ],
            "wires": []
        }
        response = self.client.post("/voltForge-ai/api/v1/model/circuit/export", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("spiceNetlist", data)
        self.assertIn("bomCsv", data)
        self.assertIn("Item,Designator,Component", data["bomCsv"])

    def test_auto_layout_circuit_endpoint(self):
        payload = {
            "boardType": "ARDUINO_UNO",
            "components": [
                {"id": "c1", "type": "RESISTOR", "x": 0, "y": 0},
                {"id": "c2", "type": "LED", "x": 500, "y": 500}
            ],
            "wires": [
                {"fromNodeId": "c1", "toNodeId": "c2"}
            ]
        }
        response = self.client.post("/voltForge-ai/api/v1/model/circuit/auto-layout", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(len(data["components"]), 2)

    def test_thermal_analysis_circuit_endpoint(self):
        payload = {
            "boardType": "ARDUINO_UNO",
            "components": [
                {"id": "r1", "type": "RESISTOR", "value": "220"},
                {"id": "reg1", "type": "VOLTAGE_REGULATOR_LM7805", "value": "5V"}
            ],
            "wires": []
        }
        response = self.client.post("/voltForge-ai/api/v1/model/circuit/thermal-analysis", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("thermalNodes", data)
        self.assertGreaterEqual(len(data["thermalNodes"]), 2)

    def test_synthesize_circuit_endpoint(self):
        payload = {
            "message": "Create an ESP32 temperature sensor with OLED display and buzzer alarm",
            "boardType": "ESP32_DEVKIT_V1"
        }
        response = self.client.post("/voltForge-ai/api/v1/model/circuit/synthesize", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("components", data)
        self.assertIn("generatedFirmware", data)
        self.assertIn("DHT", data["generatedFirmware"])

    def test_evaluate_emi_rules_endpoint(self):
        payload = {
            "boardType": "ESP32_DEVKIT_V1",
            "components": [
                {"id": "mcu", "type": "ESP32_DEVKIT_V1", "x": 100, "y": 100}
            ],
            "wires": []
        }
        response = self.client.post("/voltForge-ai/api/v1/model/circuit/emi-rules", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertGreaterEqual(data["violationCount"], 1)
        self.assertGreaterEqual(len(data["proposedInsertions"]), 1)

    def test_calculate_bom_sourcing_endpoint(self):
        payload = {
            "boardType": "ESP32_DEVKIT_V1",
            "components": [
                {"id": "r1", "type": "RESISTOR", "value": "10k"},
                {"id": "mcu", "type": "ESP32_DEVKIT_V1", "value": "DevKit"}
            ],
            "wires": []
        }
        response = self.client.post("/voltForge-ai/api/v1/model/circuit/bom-sourcing", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("estimatedUnitCost", data)
        self.assertIn("volumePricing", data)
        self.assertIn("qty_100", data["volumePricing"])


if __name__ == "__main__":
    unittest.main()





