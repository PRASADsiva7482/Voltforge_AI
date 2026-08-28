import unittest
import uuid
from api.database import db
from config import DB_ENABLED, DB_PASSWORD


@unittest.skipUnless(
    DB_ENABLED and bool(DB_PASSWORD),
    "Set VOLTFORGE_AI_DB_ENABLED=true and configure DB credentials to run integration tests.",
)
class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db = db

    def test_database_connection(self):
        health = self.db.check_health()
        self.assertTrue(health.get("connected"), f"Database connection failed: {health}")
        self.assertGreaterEqual(health.get("tableCount", 0), 18)

    def test_active_rules_retrieval(self):
        rules = self.db.get_active_rules()
        self.assertIsInstance(rules, list)
        self.assertGreater(len(rules), 0)
        rule_codes = [r["rule_code"] for r in rules]
        self.assertIn("RULE_VCC_GND_SHORT", rule_codes)

    def test_save_chat_turn(self):
        session_id = str(uuid.uuid4())
        res_session = self.db.save_chat_turn(
            session_id=session_id,
            project_id="test_project_1",
            user_id="test_user_1",
            user_message="How to connect a 555 timer astable?",
            assistant_reply="Connect pin 8 and 4 to VCC, pin 1 to GND...",
            board_type="ARDUINO_UNO",
            intent="CIRCUIT_EXPLANATION",
            confidence=0.92,
            latency_ms=120,
        )
        self.assertEqual(res_session, session_id)

    def test_save_circuit_validation(self):
        validation_id = self.db.save_circuit_validation(
            project_id="test_project_2",
            snapshot_id=None,
            is_valid=False,
            safety_score=65,
            general_feedback="Missing current limiting resistor on LED.",
            issues=[
                {
                    "severity": "CRITICAL",
                    "category": "OVERCURRENT",
                    "componentId": "led1",
                    "pinName": "anode",
                    "message": "LED connected to GPIO without series resistor.",
                    "suggestedFix": "Add a 220 Ohm resistor in series.",
                }
            ],
            duration_ms=45,
        )
        self.assertIsNotNone(validation_id)

    def test_spice_cache_roundtrip(self):
        circuit_hash = "test_hash_" + str(uuid.uuid4())[:8]
        saved = self.db.save_spice_simulation(
            circuit_hash=circuit_hash,
            analysis_type="OP_DC",
            netlist_content="V1 1 0 DC 5\nR1 1 2 220\nD1 2 0 LED",
            node_voltages={"1": 5.0, "2": 2.1},
            branch_currents={"R1": 0.013},
            power_dissipation={"R1": 0.037},
            converged=True,
            duration_ms=12,
        )
        self.assertTrue(saved)

        cached = self.db.get_cached_spice_simulation(circuit_hash, "OP_DC")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["nodeVoltages"]["1"], 5.0)
        self.assertEqual(cached["branchCurrents"]["R1"], 0.013)


if __name__ == "__main__":
    unittest.main()
