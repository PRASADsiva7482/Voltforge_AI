import unittest
from engine.digital_logic import (
    TruthTableGenerator, KarnaughMapSimplifier, IC74SeriesEmulator,
    SevenSegmentDecoder, FSMDesigner, SynchronousCounterBuilder
)
from engine.analog_engine import (
    BJTSolver, MOSFETSolver, OpAmpSolver, ZenerRegulatorSolver,
    WheatStoneBridgeSolver, NTCThermistorSolver, R2RDACCalculator,
    ADCQuantizationAnalyzer, CurrentShuntSolver
)
from engine.deep_firmware import DeepFirmwareAnalyzer
from engine.pcb_engine import (
    TrackWidthCalculator, ViaCurrentCapacity, DifferentialPairCalculator,
    PCBCostEstimator, ThermalViaArraySolver, KiCadExporter
)
from engine.context_memory import (
    ConversationMemory, EntityExtractor, IntentClassifier,
    ClarificationGenerator, AutoSuggestEngine
)
from engine.security import (
    RateLimiter, InputSanitizer, PerformanceMetrics, SecurityHeaders, HealthChecker
)
from engine.reasoning import ElectronicsReasoningOrchestrator


class TestDigitalLogicEngine(unittest.TestCase):
    def test_truth_table_and(self):
        res = TruthTableGenerator.generate("AND", 2)
        self.assertEqual(res["totalCombinations"], 4)
        self.assertEqual(res["truthTable"][3]["output"], 1)

    def test_kmap_simplifier(self):
        res = KarnaughMapSimplifier.simplify_2var([0, 1, 2, 3])
        self.assertEqual(res["simplifiedExpression"], "1")

    def test_ic_emulator(self):
        info = IC74SeriesEmulator.get_ic_info("74HC595")
        self.assertIsNotNone(info)
        self.assertEqual(info["partNumber"], "74HC595")

    def test_seven_segment(self):
        res = SevenSegmentDecoder.decode(5)
        self.assertEqual(res["digit"], 5)
        self.assertIn("a", res["activeSegments"])

    def test_counter_builder(self):
        res = SynchronousCounterBuilder.build(8, "up")
        self.assertEqual(res["flipFlopsNeeded"], 3)


class TestAnalogEngine(unittest.TestCase):
    def test_bjt_solver(self):
        res = BJTSolver.solve_common_emitter(12, 100000, 1000)
        self.assertIn("ic_mA", res)
        self.assertGreater(res["ic_mA"], 0)

    def test_mosfet_nmos(self):
        res = MOSFETSolver.solve_nmos(3.0, 5.0, 1.5)
        self.assertEqual(res["region"], "Saturation")

    def test_opamp_inverting(self):
        res = OpAmpSolver.inverting_amplifier(100000, 10000, 1.0)
        self.assertEqual(res["closedLoopGain"], -10.0)

    def test_zener_regulator(self):
        res = ZenerRegulatorSolver.design(12, 5.1, 20)
        self.assertTrue(res["isSafe"])

    def test_ntc_thermistor(self):
        res = NTCThermistorSolver.calculate_temperature(10000)
        self.assertAlmostEqual(res["temperature_C"], 25.0, delta=1.0)

    def test_r2r_dac(self):
        res = R2RDACCalculator.calculate(128, 8, 5.0)
        self.assertAlmostEqual(res["vOut"], 2.5, delta=0.1)

    def test_adc_quantization(self):
        res = ADCQuantizationAnalyzer.analyze(12, 3.3)
        self.assertEqual(res["quantizationLevels"], 4096)


class TestDeepFirmwareAnalyzer(unittest.TestCase):
    def test_memory_usage(self):
        code = "#include <Arduino.h>\nint x = 10;\nvoid setup() {}\nvoid loop() {}"
        res = DeepFirmwareAnalyzer.analyze_memory_usage(code, "ARDUINO_UNO")
        self.assertGreater(res["estimatedSRAM_bytes"], 0)

    def test_race_conditions(self):
        code = "int counter = 0;\nvoid IRAM_ATTR isr() { counter++; }\nvoid setup() {}\nvoid loop() {}"
        res = DeepFirmwareAnalyzer.detect_race_conditions(code)
        self.assertGreater(res["issueCount"], 0)

    def test_deep_sleep_config(self):
        res = DeepFirmwareAnalyzer.generate_deep_sleep_config("ESP32", "timer", 60)
        self.assertIn("esp_deep_sleep_start", res["generatedCode"])


class TestPCBEngine(unittest.TestCase):
    def test_track_width(self):
        res = TrackWidthCalculator.calculate_external(2.0, 10.0, 1.0)
        self.assertGreater(res["requiredWidth_mm"], 0)

    def test_via_current(self):
        res = ViaCurrentCapacity.calculate(0.3, 1.0, 4)
        self.assertGreater(res["totalCurrentCapacity_A"], 0)

    def test_pcb_cost(self):
        res = PCBCostEstimator.estimate(50, 50, 2, 10)
        self.assertGreater(res["estimatedTotal_USD"], 0)


class TestContextMemory(unittest.TestCase):
    def test_entity_extractor(self):
        entities = EntityExtractor.extract("Connect 5V LED to pin 13 on ESP32")
        self.assertIn(5.0, entities.get("voltages", []))
        self.assertEqual(entities.get("board"), "ESP32")

    def test_intent_classifier(self):
        res = IntentClassifier.classify("My OLED display is blank and not working")
        self.assertEqual(res["intent"], "troubleshoot")

    def test_conversation_memory(self):
        mem = ConversationMemory()
        mem.add_turn("user", "Build a weather station with ESP32", "design")
        summary = mem.get_context_summary()
        self.assertEqual(summary["turnCount"], 1)


class TestSecurity(unittest.TestCase):
    def test_rate_limiter(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        self.assertTrue(limiter.is_allowed("client1")["allowed"])
        self.assertTrue(limiter.is_allowed("client1")["allowed"])
        self.assertFalse(limiter.is_allowed("client1")["allowed"])

    def test_input_sanitizer(self):
        res = InputSanitizer.sanitize_message("<script>alert('test')</script>")
        self.assertFalse(res["valid"])

    def test_health_check(self):
        res = HealthChecker.check()
        self.assertIn("status", res)


class TestReasoningOrchestratorV2(unittest.TestCase):
    def setUp(self):
        self.orchestrator = ElectronicsReasoningOrchestrator()

    def test_digital_logic_dispatch(self):
        resp = self.orchestrator.process_chat("Show me truth table for NAND gate")
        self.assertIn("Truth Table", resp.reply)

    def test_analog_dispatch(self):
        resp = self.orchestrator.process_chat("Calculate BJT common emitter bias point")
        self.assertIn("BJT Common-Emitter", resp.reply)

    def test_pcb_dispatch(self):
        resp = self.orchestrator.process_chat("What is the PCB trace width for 2A?")
        self.assertIn("Trace Width", resp.reply)


if __name__ == "__main__":
    unittest.main()
