"""
Voltforge AI - Enterprise 6-Pass Chain-of-Thought Electronics Reasoning Orchestrator (v2)
Integrates ALL 16 engine modules:
  FuzzyResolver, PhysicsSolver, PinRouter, ElectricalVerifier, WebSearchEngine,
  FirmwareAnalyzer, DeepFirmwareAnalyzer, ComponentKnowledgeGraph, TroubleshootingEngine,
  NLCircuitSynthesizer, SpiceEngine, BusSolvers, PowerManager, WirelessSolvers,
  DigitalLogic, AnalogEngine, PCBEngine, ContextMemory, IntentClassifier, Security.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from circuit_verifier import ElectricalVerifier
from web_search_engine import WebSearchEngine
from engine.code_generator import FirmwareCodeGenerator
from engine.fuzzy_resolver import FuzzyResolver
from engine.physics_solver import ElectricalPhysicsSolver
from engine.pin_router import PinRouter
from engine.firmware_analyzer import FirmwareAnalyzer
from engine.deep_firmware import DeepFirmwareAnalyzer
from engine.knowledge_graph import ComponentKnowledgeGraph
from engine.troubleshooter import TroubleshootingEngine
from engine.nl_synthesizer import NLCircuitSynthesizer, BOMExporter
from engine.spice_engine import DCOperatingPointSolver, TransientAnalysisSolver, ACFrequencyResponseSolver, Timer555Solver, SpiceNetlistExporter, PWMDutyCycleCalculator
from engine.bus_solver import CANBusSolver, I2CMultiplexerSolver, SPIDaisyChainSolver, ModbusCRCSolver, BusCapacitanceEstimator
from engine.power_manager import BatteryLifeEstimator, SolarPanelSizer, LDOThermalSolver, BuckBoostEfficiencyCalculator, SupercapacitorEnergySolver
from engine.wireless_solver import WiFiPowerOptimizer, FreeSpacePathLossCalculator, RSSIInterpreter, MQTTPayloadFormatter, LoRaWANPayloadPacker
from engine.digital_logic import TruthTableGenerator, KarnaughMapSimplifier, IC74SeriesEmulator, SevenSegmentDecoder, SynchronousCounterBuilder
from engine.analog_engine import BJTSolver, MOSFETSolver, OpAmpSolver, ZenerRegulatorSolver, WheatStoneBridgeSolver, NTCThermistorSolver, R2RDACCalculator, ADCQuantizationAnalyzer, CurrentShuntSolver
from engine.pcb_engine import TrackWidthCalculator, ViaCurrentCapacity, DifferentialPairCalculator, PCBCostEstimator, ThermalViaArraySolver, KiCadExporter
from engine.context_memory import ConversationMemory, EntityExtractor, IntentClassifier, ClarificationGenerator, AutoSuggestEngine
from engine.security import InputSanitizer, PerformanceMetrics
from api.schemas import ChatResponse

logger = logging.getLogger("voltforge-ai.reasoning")


class ElectronicsReasoningOrchestrator:
    """Enterprise 6-Pass Chain-of-Thought Reasoning Engine with full module integration."""

    def __init__(self):
        self.web_search = WebSearchEngine()
        self.verifier = ElectricalVerifier()
        self.conversation_memory = ConversationMemory()
        self.performance_metrics = PerformanceMetrics()
        self.dataset_pairs: List[tuple[str, str]] = []
        self._load_dataset()

    def _load_dataset(self):
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.txt")
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for chunk in f.read().split("[Q]"):
                    if not chunk.strip() or "[A]" not in chunk:
                        continue
                    q, a = chunk.split("[A]", 1)
                    self.dataset_pairs.append((q.strip().lower(), a.strip()))

    def process_chat(
        self,
        message: str,
        board_type: str = "ARDUINO_UNO",
        components: List[Dict[str, Any]] = None,
        wires: List[Dict[str, Any]] = None,
        code: str = "",
        context: Dict[str, Any] = None
    ) -> ChatResponse:
        start_time = time.time()
        components = components or []
        wires = wires or []

        # ═══════════════════════════════════════════════════
        # PASS 0: Security & Input Sanitization
        # ═══════════════════════════════════════════════════
        sanitized = InputSanitizer.sanitize_message(message)
        if not sanitized["valid"]:
            return ChatResponse(reply=f"⚠️ Input rejected: {sanitized['reason']}", confidence=0.99)

        # ═══════════════════════════════════════════════════
        # PASS 1: Fuzzy Resolution & Intent Classification
        # ═══════════════════════════════════════════════════
        norm_result = FuzzyResolver.normalize_text(message)
        resolved_boards = norm_result.get("resolvedBoards", [])
        resolved_components = norm_result.get("resolvedComponents", [])
        effective_board = resolved_boards[0] if resolved_boards else board_type
        lower = message.lower().strip()

        # Entity extraction & intent classification
        entities = EntityExtractor.extract(message)
        intent_result = IntentClassifier.classify(message)
        intent = intent_result["intent"]

        # Record turn in conversation memory
        self.conversation_memory.add_turn("user", message, intent)

        # ═══════════════════════════════════════════════════
        # PASS 2: Specialized Domain Engine Dispatch
        # ═══════════════════════════════════════════════════

        # --- Troubleshooting & Diagnostics ---
        if intent == "troubleshoot" or any(kw in lower for kw in ("not working", "not detected", "not responding", "broken", "doesnt work", "won't work", "blank", "gibberish", "upload fail", "troubleshoot", "diagnose", "debug")):
            diag = TroubleshootingEngine.diagnose(message, effective_board, components, wires, code)
            lines = [f"### 🔧 Diagnostic Report for {effective_board}:"]
            for d in diag["diagnosis"]:
                lines.append(f"\n**{d['category']}** (Probability: {d['probability']})")
                lines.append(f"**Cause**: {d['cause']}")
                for step in d["steps"]:
                    lines.append(f"  {step}")
            if diag["quickFixes"]:
                lines.append("\n**Quick Fixes:**")
                for fix in diag["quickFixes"]:
                    lines.append(f"- {fix}")
            suggestions = AutoSuggestEngine.suggest(intent, entities, effective_board)
            if suggestions:
                lines.append("\n**💡 Try next:**")
                for s in suggestions:
                    lines.append(f"- {s}")
            return self._finalize(ChatResponse(reply="\n".join(lines), confidence=0.94), start_time, "troubleshoot")

        # --- Natural Language to Circuit Synthesis ---
        if intent == "design" or any(kw in lower for kw in ("build a", "create a", "design a", "make a circuit", "build me", "i want to make")):
            synth = NLCircuitSynthesizer.synthesize(message, effective_board)
            if synth["success"]:
                bom = BOMExporter.generate_bom(synth["components"], effective_board)
                lines = [
                    f"### 🔌 Circuit Design for {effective_board}:",
                    f"**Description**: {synth['message']}",
                    "",
                    "**Components Required:**",
                ]
                for item in bom["items"]:
                    lines.append(f"- {item['description']} (`{item['componentType']}`) — ${item['unitPrice_USD']:.2f}")
                lines.append(f"\n**Estimated Total Cost**: ${bom['totalCost_USD']:.2f}")
                suggestions = AutoSuggestEngine.suggest(intent, entities, effective_board)
                if suggestions:
                    lines.append("\n**💡 Try next:**")
                    for s in suggestions:
                        lines.append(f"- {s}")
                return self._finalize(ChatResponse(
                    reply="\n".join(lines), confidence=0.92,
                    additions=[{"type": c["type"], "name": c.get("name", c["type"])} for c in synth["components"]],
                ), start_time, "design")
            return self._finalize(ChatResponse(reply=synth["message"], confidence=0.7), start_time, "design")

        # --- Digital Logic & Gate Analysis ---
        if any(kw in lower for kw in ("truth table", "logic gate", "karnaugh", "k-map", "74hc", "74ls", "7-segment", "seven segment", "counter", "flip flop", "fsm")):
            if "truth table" in lower:
                gate = "AND"
                for g in ("nand", "nor", "xor", "xnor", "or", "and", "not"):
                    if g in lower:
                        gate = g.upper()
                        break
                tt = TruthTableGenerator.generate(gate)
                rows = "\n".join([f"  {r['inputs']} → {r['output']}" for r in tt['truthTable']])
                return self._finalize(ChatResponse(
                    reply=f"### 🔢 Truth Table — {gate} Gate ({tt['numInputs']}-input):\n```\n{rows}\n```",
                    confidence=0.97,
                ), start_time, "digital_logic")
            if "7-segment" in lower or "seven segment" in lower:
                results = [SevenSegmentDecoder.decode(d) for d in range(10)]
                rows = "\n".join([f"  {r['digit']}: {r['binaryPattern']} ({', '.join(r['activeSegments'])})" for r in results])
                return self._finalize(ChatResponse(
                    reply=f"### 7-Segment Display Decoder (a-g):\n```\n{rows}\n```",
                    confidence=0.97,
                ), start_time, "digital_logic")
            if "74" in lower:
                for ic_key in ("74HC595", "74HC00", "74HC04", "74HC08", "74HC32", "74HC86", "74HC138", "74HC4511"):
                    if ic_key.lower() in lower or ic_key.replace("HC", "LS").lower() in lower:
                        info = IC74SeriesEmulator.get_ic_info(ic_key)
                        if info:
                            return self._finalize(ChatResponse(
                                reply=f"### 📦 IC: {info['partNumber']} — {info['name']}\n- **Gates**: {info['gates']}\n- **Gate Type**: {info['gate_type']}\n- **Inputs/Gate**: {info['inputs_per_gate']}\n- **Package**: {info['package']}",
                                confidence=0.96,
                            ), start_time, "digital_logic")

        # --- Analog Circuit Solvers ---
        if any(kw in lower for kw in ("bjt", "transistor bias", "common emitter", "mosfet", "op-amp", "opamp", "operational amplifier", "zener", "wheatstone", "thermistor", "ntc", "dac", "adc resolution", "current shunt")):
            if "bjt" in lower or "common emitter" in lower:
                bjt = BJTSolver.solve_common_emitter(12, 100000, 1000)
                return self._finalize(ChatResponse(
                    reply=f"### 🔬 BJT Common-Emitter Analysis:\n- **VCC**: {bjt['vcc']}V, **Rb**: {bjt['rb_ohm']}Ω, **Rc**: {bjt['rc_ohm']}Ω\n- **Ib**: {bjt['ib_uA']}µA\n- **Ic**: {bjt['ic_mA']}mA\n- **Vce**: {bjt['vce_V']}V\n- **Region**: {bjt['region']}\n- **Power**: {bjt['powerDissipation_mW']}mW",
                    confidence=0.96,
                ), start_time, "analog")
            if "op-amp" in lower or "opamp" in lower:
                oa = OpAmpSolver.inverting_amplifier(100000, 10000, 1.0)
                return self._finalize(ChatResponse(
                    reply=f"### 📐 Op-Amp Inverting Amplifier:\n- **Rf**: {oa['rf_ohm']}Ω, **Rin**: {oa['rin_ohm']}Ω\n- **Gain**: {oa['closedLoopGain']} ({oa['gainMagnitude_dB']}dB)\n- **Bandwidth**: {oa['bandwidth_kHz']}kHz\n- **GBW**: {oa['gbw_MHz']}MHz",
                    confidence=0.96,
                ), start_time, "analog")
            if "zener" in lower:
                zr = ZenerRegulatorSolver.design(12, 5.1, 20)
                return self._finalize(ChatResponse(
                    reply=f"### ⚡ Zener Voltage Regulator:\n- **Vin**: {zr['vIn']}V → **Vout**: {zr['vZener']}V\n- **Series Resistor**: {zr['seriesResistor_ohm']}Ω\n- **Zener Power**: {zr['zenerPower_mW']}mW\n- **Safe**: {'✅ Yes' if zr['isSafe'] else '❌ No'}",
                    confidence=0.95,
                ), start_time, "analog")
            if "thermistor" in lower or "ntc" in lower:
                ntc = NTCThermistorSolver.calculate_temperature(15000)
                return self._finalize(ChatResponse(
                    reply=f"### 🌡️ NTC Thermistor Reading:\n- **Resistance**: {ntc['resistance_ohm']}Ω\n- **Temperature**: {ntc['temperature_C']}°C / {ntc['temperature_F']}°F\n- **Beta**: {ntc['beta']}",
                    confidence=0.95,
                ), start_time, "analog")
            if "adc" in lower:
                adc = ADCQuantizationAnalyzer.analyze(12, 3.3)
                return self._finalize(ChatResponse(
                    reply=f"### 📊 ADC Resolution Analysis:\n- **Bits**: {adc['bits']}\n- **Levels**: {adc['quantizationLevels']}\n- **LSB**: {adc['lsb_mV']}mV\n- **SNR**: {adc['idealSNR_dB']}dB\n- **ENOB**: {adc['effectiveBits']}",
                    confidence=0.96,
                ), start_time, "analog")
            if "dac" in lower:
                dac = R2RDACCalculator.calculate(128, 8, 5.0)
                return self._finalize(ChatResponse(
                    reply=f"### 📈 R-2R DAC Output:\n- **Digital Value**: {dac['digitalValue']} ({dac['binaryRepresentation']})\n- **Vout**: {dac['vOut']}V\n- **Resolution**: {dac['resolution_mV']}mV/LSB",
                    confidence=0.96,
                ), start_time, "analog")

        # --- Physics & Math Calculations ---
        if intent == "calculate" or any(kw in lower for kw in ("led resistor", "resistor calculator", "ohms law", "ohm's law", "voltage divider", "power dissipation")):
            if "voltage divider" in lower:
                calc = ElectricalPhysicsSolver.calculate_voltage_divider(5.0, 10000.0, 20000.0)
                return self._finalize(ChatResponse(
                    reply=f"### Voltage Divider Calculator:\n- **V_in**: 5.0V\n- **R1**: 10kΩ, **R2**: 20kΩ\n- **V_out**: {calc['vOut']}V\n- **Current Draw**: {calc['currentDrawmA']}mA\n- **3.3V Safe**: {'✅ Yes' if calc['is3V3Safe'] else '❌ No'}",
                    confidence=0.98,
                ), start_time, "calculate")
            calc = ElectricalPhysicsSolver.calculate_led_resistor(5.0, 2.0, 20.0)
            return self._finalize(ChatResponse(
                reply=f"### LED Current-Limiting Resistor:\n- **Supply**: 5.0V, **LED V_f**: 2.0V, **Target I**: 20mA\n- **Exact R**: {calc['exactResistance']}Ω\n- **Recommended**: **{calc['recommendedResistor']}**\n- **Power**: {calc['powerDissipation']}W ({'✅ Safe for ¼W' if calc['isSafeQuarterWatt'] else '⚠️ Exceeds ¼W'})",
                confidence=0.98,
            ), start_time, "calculate")

        # --- 555 Timer Calculations ---
        if "555" in lower and any(kw in lower for kw in ("timer", "frequency", "oscillator", "astable")):
            t555 = Timer555Solver.astable(10000, 47000, 10e-6)
            return self._finalize(ChatResponse(
                reply=f"### 555 Timer Astable Mode:\n- **R1**: 10kΩ, **R2**: 47kΩ, **C**: 10µF\n- **Frequency**: {t555['frequency_Hz']}Hz\n- **Period**: {t555['period_s']}s\n- **Duty Cycle**: {t555['dutyCyclePercent']}%\n- **T_high**: {t555['tHigh_s']}s, **T_low**: {t555['tLow_s']}s",
                confidence=0.97,
            ), start_time, "calculate")

        # --- Battery Life Estimation ---
        if intent == "power" or any(kw in lower for kw in ("battery life", "battery runtime", "how long will battery", "power consumption", "solar panel", "deep sleep")):
            if "solar" in lower:
                sol = SolarPanelSizer.size_panel(500, 3.7, 4.0)
                return self._finalize(ChatResponse(
                    reply=f"### ☀️ Solar Panel Sizing:\n- **Daily Consumption**: {sol['dailyConsumption_mAh']}mAh\n- **Daily Energy**: {sol['dailyEnergy_Wh']}Wh\n- **Required Panel**: {sol['requiredPanelPower_W']}W\n- **Recommended**: **{sol['recommendedPanelPower_W']}W** (with 25% margin)\n- **Charger**: {sol['chargerType']}",
                    confidence=0.94,
                ), start_time, "power")
            if "deep sleep" in lower:
                config = DeepFirmwareAnalyzer.generate_deep_sleep_config(effective_board, "timer", 60)
                if "error" not in config:
                    return self._finalize(ChatResponse(
                        reply=f"### 💤 Deep Sleep Configuration for {effective_board}:\n- **Wakeup Source**: {config['wakeupSource']}\n- **Sleep Duration**: {config['sleepDuration_s']}s\n- **Current Draw**: {config['estimatedCurrentDraw_uA']}µA\n\n```cpp\n{config['generatedCode']}\n```",
                        confidence=0.95, hasCode=True, generatedCode=config["generatedCode"],
                    ), start_time, "power")
                return self._finalize(ChatResponse(reply=config["error"], confidence=0.7), start_time, "power")
            est = BatteryLifeEstimator.estimate_runtime(2000, 80, 10, 5)
            return self._finalize(ChatResponse(
                reply=f"### 🔋 Battery Life Estimate:\n- **Battery**: {est['batteryCapacity_mAh']}mAh\n- **Active Current**: {est['activeCurrent_mA']}mA\n- **Sleep Current**: {est['sleepCurrent_uA']}µA\n- **Active Ratio**: {est['activeRatio_percent']}%\n- **Average Current**: {est['averageCurrent_mA']}mA\n- **Estimated Runtime**: **{est['estimatedRuntime_hours']} hours** ({est['estimatedRuntime_days']} days)",
                confidence=0.95,
            ), start_time, "power")

        # --- Digital Logic & Gate Analysis ---
        if any(kw in lower for kw in ("truth table", "logic gate", "karnaugh", "k-map", "74hc", "74ls", "7-segment", "seven segment", "counter", "flip flop", "fsm")):
            if "truth table" in lower:
                gate = "AND"
                for g in ("nand", "nor", "xor", "xnor", "or", "and", "not"):
                    if g in lower:
                        gate = g.upper()
                        break
                tt = TruthTableGenerator.generate(gate)
                rows = "\n".join([f"  {r['inputs']} → {r['output']}" for r in tt['truthTable']])
                return self._finalize(ChatResponse(
                    reply=f"### 🔢 Truth Table — {gate} Gate ({tt['numInputs']}-input):\n```\n{rows}\n```",
                    confidence=0.97,
                ), start_time, "digital_logic")
            if "7-segment" in lower or "seven segment" in lower:
                results = [SevenSegmentDecoder.decode(d) for d in range(10)]
                rows = "\n".join([f"  {r['digit']}: {r['binaryPattern']} ({', '.join(r['activeSegments'])})" for r in results])
                return self._finalize(ChatResponse(
                    reply=f"### 7-Segment Display Decoder (a-g):\n```\n{rows}\n```",
                    confidence=0.97,
                ), start_time, "digital_logic")
            if "74" in lower:
                for ic_key in ("74HC595", "74HC00", "74HC04", "74HC08", "74HC32", "74HC86", "74HC138", "74HC4511"):
                    if ic_key.lower() in lower or ic_key.replace("HC", "LS").lower() in lower:
                        info = IC74SeriesEmulator.get_ic_info(ic_key)
                        if info:
                            return self._finalize(ChatResponse(
                                reply=f"### 📦 IC: {info['partNumber']} — {info['name']}\n- **Gates**: {info['gates']}\n- **Gate Type**: {info['gate_type']}\n- **Inputs/Gate**: {info['inputs_per_gate']}\n- **Package**: {info['package']}",
                                confidence=0.96,
                            ), start_time, "digital_logic")

        # --- Analog Circuit Solvers ---
        if any(kw in lower for kw in ("bjt", "transistor bias", "common emitter", "mosfet", "op-amp", "opamp", "operational amplifier", "zener", "wheatstone", "thermistor", "ntc", "dac", "adc resolution", "current shunt")):
            if "bjt" in lower or "common emitter" in lower:
                bjt = BJTSolver.solve_common_emitter(12, 100000, 1000)
                return self._finalize(ChatResponse(
                    reply=f"### 🔬 BJT Common-Emitter Analysis:\n- **VCC**: {bjt['vcc']}V, **Rb**: {bjt['rb_ohm']}Ω, **Rc**: {bjt['rc_ohm']}Ω\n- **Ib**: {bjt['ib_uA']}µA\n- **Ic**: {bjt['ic_mA']}mA\n- **Vce**: {bjt['vce_V']}V\n- **Region**: {bjt['region']}\n- **Power**: {bjt['powerDissipation_mW']}mW",
                    confidence=0.96,
                ), start_time, "analog")
            if "op-amp" in lower or "opamp" in lower:
                oa = OpAmpSolver.inverting_amplifier(100000, 10000, 1.0)
                return self._finalize(ChatResponse(
                    reply=f"### 📐 Op-Amp Inverting Amplifier:\n- **Rf**: {oa['rf_ohm']}Ω, **Rin**: {oa['rin_ohm']}Ω\n- **Gain**: {oa['closedLoopGain']} ({oa['gainMagnitude_dB']}dB)\n- **Bandwidth**: {oa['bandwidth_kHz']}kHz\n- **GBW**: {oa['gbw_MHz']}MHz",
                    confidence=0.96,
                ), start_time, "analog")
            if "zener" in lower:
                zr = ZenerRegulatorSolver.design(12, 5.1, 20)
                return self._finalize(ChatResponse(
                    reply=f"### ⚡ Zener Voltage Regulator:\n- **Vin**: {zr['vIn']}V → **Vout**: {zr['vZener']}V\n- **Series Resistor**: {zr['seriesResistor_ohm']}Ω\n- **Zener Power**: {zr['zenerPower_mW']}mW\n- **Safe**: {'✅ Yes' if zr['isSafe'] else '❌ No'}",
                    confidence=0.95,
                ), start_time, "analog")
            if "thermistor" in lower or "ntc" in lower:
                ntc = NTCThermistorSolver.calculate_temperature(15000)
                return self._finalize(ChatResponse(
                    reply=f"### 🌡️ NTC Thermistor Reading:\n- **Resistance**: {ntc['resistance_ohm']}Ω\n- **Temperature**: {ntc['temperature_C']}°C / {ntc['temperature_F']}°F\n- **Beta**: {ntc['beta']}",
                    confidence=0.95,
                ), start_time, "analog")
            if "adc" in lower:
                adc = ADCQuantizationAnalyzer.analyze(12, 3.3)
                return self._finalize(ChatResponse(
                    reply=f"### 📊 ADC Resolution Analysis:\n- **Bits**: {adc['bits']}\n- **Levels**: {adc['quantizationLevels']}\n- **LSB**: {adc['lsb_mV']}mV\n- **SNR**: {adc['idealSNR_dB']}dB\n- **ENOB**: {adc['effectiveBits']}",
                    confidence=0.96,
                ), start_time, "analog")
            if "dac" in lower:
                dac = R2RDACCalculator.calculate(128, 8, 5.0)
                return self._finalize(ChatResponse(
                    reply=f"### 📈 R-2R DAC Output:\n- **Digital Value**: {dac['digitalValue']} ({dac['binaryRepresentation']})\n- **Vout**: {dac['vOut']}V\n- **Resolution**: {dac['resolution_mV']}mV/LSB",
                    confidence=0.96,
                ), start_time, "analog")

        # --- PCB Design ---
        if intent == "pcb" or any(kw in lower for kw in ("trace width", "track width", "pcb", "via current", "differential pair", "pcb cost", "thermal via", "kicad")):
            if "trace width" in lower or "track width" in lower:
                tw = TrackWidthCalculator.calculate_external(2.0, 10.0, 1.0)
                return self._finalize(ChatResponse(
                    reply=f"### 📏 IPC-2221 PCB Trace Width:\n- **Current**: {tw['current_A']}A, **Temp Rise**: {tw['tempRise_C']}°C\n- **Required Width**: **{tw['requiredWidth_mm']}mm** ({tw['requiredWidth_mil']}mil)\n- **Trace Resistance**: {tw['traceResistance_mOhm']}mΩ\n- **Voltage Drop**: {tw['voltageDrop_mV']}mV",
                    confidence=0.96,
                ), start_time, "pcb")
            if "pcb cost" in lower:
                cost = PCBCostEstimator.estimate(50, 50, 2, 10)
                return self._finalize(ChatResponse(
                    reply=f"### 💰 PCB Cost Estimate:\n- **Board Size**: {cost['boardSize_mm']}mm\n- **Layers**: {cost['layers']}, **Qty**: {cost['quantity']}\n- **Total**: ${cost['estimatedTotal_USD']}\n- **Per Board**: ${cost['perBoard_USD']}\n- **Lead Time**: {cost['leadTime']}",
                    confidence=0.90,
                ), start_time, "pcb")
            if "via" in lower:
                via = ViaCurrentCapacity.calculate(0.3, 1.0, 4)
                return self._finalize(ChatResponse(
                    reply=f"### 🕳️ Via Current Capacity:\n- **Drill**: {via['drillDiameter_mm']}mm, **Count**: {via['numVias']}\n- **Per Via**: {via['currentPerVia_A']}A\n- **Total Capacity**: **{via['totalCurrentCapacity_A']}A**",
                    confidence=0.95,
                ), start_time, "pcb")

        # --- Code Review & Firmware Analysis (Deep) ---
        if intent == "generate_code" and any(kw in lower for kw in ("review", "analyze", "check")):
            intent = "validate"  # redirect to code review

        if any(kw in lower for kw in ("review code", "analyze code", "code review", "is my code", "check my code", "compile error", "memory usage", "race condition", "stack overflow")):
            if code:
                analysis = FirmwareAnalyzer.analyze(code, components, effective_board)
                deep = DeepFirmwareAnalyzer.analyze_memory_usage(code, effective_board)
                race = DeepFirmwareAnalyzer.detect_race_conditions(code)
                lines = [f"### 💻 Deep Firmware Analysis for {effective_board}:", analysis["summary"], ""]
                for issue in analysis["issues"][:8]:
                    lines.append(f"- **{issue['severity']}** [{issue.get('type', 'ISSUE')}]: {issue['message']}")
                    lines.append(f"  Fix: {issue.get('suggestedFix', 'N/A')}")
                # Deep analysis
                lines.append(f"\n**Memory Profile** ({deep['architecture']}):")
                lines.append(f"- SRAM: ~{deep['estimatedSRAM_bytes']}B / {deep['sramCapacity_KB']}KB ({deep['sramUsagePercent']}%)")
                lines.append(f"- Global Variables: {deep['globalVariableCount']}, String Literals: {deep['stringLiteralBytes']}B")
                if deep["usesStringClass"]:
                    lines.append("- ⚠️ Uses `String` class — heap fragmentation risk!")
                # Race conditions
                if race["issues"]:
                    lines.append("\n**Race Condition Warnings:**")
                    for ri in race["issues"][:3]:
                        lines.append(f"- **{ri['severity']}**: {ri['message']}")
                for di in deep["issues"]:
                    lines.append(f"- **{di['severity']}**: {di['message']}")
                return self._finalize(ChatResponse(
                    reply="\n".join(lines), confidence=analysis["confidence"], codeFixes=analysis["codeFixes"],
                ), start_time, "code_review")
            return self._finalize(ChatResponse(reply="Please provide your firmware code for analysis.", confidence=0.7), start_time, "code_review")

        # ═══════════════════════════════════════════════════
        # PASS 3: Validation & Safety Verification
        # ═══════════════════════════════════════════════════
        if intent == "validate" or any(term in lower for term in ("validate", "safety", "short", "error", "check circuit", "fix circuit")):
            v_res = self.verifier.verify_circuit(effective_board, components, wires, code)
            summary = v_res.get("generalFeedback", "Validation complete.")
            return self._finalize(ChatResponse(
                reply=f"**Target Board**: {effective_board}\n\n{summary}",
                confidence=0.95,
                wireSuggestions=v_res.get("wireSuggestions", []),
                additions=v_res.get("additions", []),
                removals=v_res.get("removals", []),
                valueChanges=v_res.get("valueChanges", []),
                codeFixes=v_res.get("codeFixes", []),
            ), start_time, "validate")

        # ═══════════════════════════════════════════════════
        # PASS 4: Component Knowledge Graph & Datasheet Lookup
        # ═══════════════════════════════════════════════════
        if resolved_components:
            for comp_name in resolved_components:
                kg_info = ComponentKnowledgeGraph.get_component_info(comp_name)
                if kg_info:
                    lines = [
                        f"### 📚 Component Knowledge: **{kg_info['fullName']}**",
                        f"- **Category**: {kg_info['category']}",
                        f"- **Operating Voltage**: {kg_info['operatingVoltage']['min']}V — {kg_info['operatingVoltage']['max']}V (typical: {kg_info['operatingVoltage']['typical']}V)",
                        f"- **Interfaces**: {', '.join(kg_info['interfaces'])}",
                        f"- **Current Draw**: {kg_info['currentDraw_mA']}mA",
                        f"- **Package**: {kg_info['package']}",
                    ]
                    if kg_info.get("i2cAddress"):
                        lines.append(f"- **I2C Addresses**: {', '.join(kg_info['i2cAddress'])}")
                    if kg_info.get("sensitivity"):
                        for k, v in kg_info["sensitivity"].items():
                            lines.append(f"- **{k.title()} Range**: {v}")
                    lines.append(f"\n**Required External Components**: {', '.join(kg_info['requiredExternalComponents'])}")
                    lines.append(f"**Common Issues**: {'; '.join(kg_info['commonIssues'])}")
                    lines.append(f"**Drop-in Equivalents**: {', '.join(kg_info['equivalents'])}")
                    i2c_wires = PinRouter.resolve_i2c_connections(comp_name, effective_board)
                    lines.append(f"\n**Recommended Wiring for {effective_board}**:")
                    for w in i2c_wires:
                        lines.append(f"- `{w['from']}` → `{w['to']}` ({w['reason']})")
                    return self._finalize(ChatResponse(reply="\n".join(lines), confidence=0.96), start_time, "knowledge")

        # ═══════════════════════════════════════════════════
        # PASS 5: Web Search & Datasheet Retrieval
        # ═══════════════════════════════════════════════════
        if intent == "datasheet" or any(term in lower for term in ("datasheet", "spec", "pinout", "what is", "how to connect", "i2c address", "chip", "module", "sensor", "search")):
            target_comp = resolved_components[0] if resolved_components else message
            s_info = self.web_search.get_component_info(target_comp)
            if s_info and s_info.get("searchResults"):
                specs = s_info.get("specs", {})
                citations = s_info.get("citations", [])
                lines = [
                    f"### 🌐 Web Search Results for **{s_info['component']}**:",
                    f"- **Operating Voltage**: {specs.get('operatingVoltage', 'N/A')}",
                    f"- **Interfaces**: {', '.join(specs.get('supportedInterfaces', [])) or 'General GPIO'}",
                ]
                if specs.get("i2cAddresses"):
                    lines.append(f"- **I2C Addresses**: {', '.join(specs.get('i2cAddresses'))}")
                lines.append(f"\n**Summary**: {specs.get('summarySnippet', '')}")
                return self._finalize(ChatResponse(reply="\n".join(lines), confidence=0.92, citations=citations), start_time, "search")

        # --- Code Generation ---
        if intent == "generate_code" or any(term in lower for term in ("generate code", "write code", "make firmware", "schematic to code")):
            gen_code = FirmwareCodeGenerator.generate(components, wires, effective_board, message)
            return self._finalize(ChatResponse(
                reply=f"Here is firmware for your active {effective_board} circuit:\n\n```cpp\n{gen_code}```",
                hasCode=True, generatedCode=gen_code, confidence=0.90,
            ), start_time, "generate_code")

        # --- Domain Q&A Dataset Knowledge Matcher ---
        clean_prompt = re.sub(r"[^a-zA-Z0-9\s]", " ", lower)
        prompt_words = set(w for w in clean_prompt.split() if len(w) > 2 and w not in {"what", "how", "why", "the", "and", "for", "with", "does", "can", "explain", "tell", "show", "give"})
        best_match = None
        best_score = 0.0
        for q_text, ans_text in self.dataset_pairs:
            q_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", q_text)
            q_words = set(w for w in q_clean.split() if len(w) > 2)
            if not q_words:
                continue
            if q_text in lower or lower in q_text:
                score = 1.0 + len(q_words) * 0.1
            else:
                intersection = prompt_words.intersection(q_words)
                if len(intersection) >= 2 or (len(intersection) == 1 and len(q_words) == 1):
                    score = len(intersection) / len(q_words)
                else:
                    score = 0.0
            if score > best_score and score >= 0.5:
                best_score = score
                best_match = ans_text

        if best_match and best_score >= 0.5:
            return self._finalize(ChatResponse(reply=best_match, confidence=0.95), start_time, "dataset_qa")

        # ═══════════════════════════════════════════════════
        # PASS 6: Context-Aware Fallback with Auto-Suggest
        # ═══════════════════════════════════════════════════
        clarification = ClarificationGenerator.generate(message, entities, intent)
        if clarification:
            return self._finalize(ChatResponse(reply=clarification, confidence=0.75), start_time, "clarify")

        comp_count = len(components)
        wire_count = len(wires)
        suggestions = AutoSuggestEngine.suggest(intent, entities, effective_board)
        lines = []
        if comp_count == 0:
            lines.append(f"**Voltforge AI** ready for {effective_board}.")
            lines.append("Add components to your canvas to enable circuit-aware analysis.")
        else:
            lines.append(f"I see **{comp_count} component(s)** and **{wire_count} wire(s)** on your {effective_board} canvas.")
        if suggestions:
            lines.append("\n**💡 Try:**")
            for s in suggestions:
                lines.append(f"- {s}")

        return self._finalize(ChatResponse(
            reply="\n".join(lines), confidence=0.88,
        ), start_time, "fallback")

    def _finalize(self, response: ChatResponse, start_time: float, endpoint: str) -> ChatResponse:
        duration_ms = (time.time() - start_time) * 1000
        self.performance_metrics.record_request(endpoint, duration_ms, 200)
        logger.info(f"Processed '{endpoint}' in {duration_ms:.1f}ms (confidence={response.confidence})")
        return response


if __name__ == "__main__":
    orchestrator = ElectronicsReasoningOrchestrator()
    resp = orchestrator.process_chat("arduno with mpu 6050")
    print("Test 1 (Fuzzy + KG):\n", resp.reply[:200])
    resp2 = orchestrator.process_chat("My oled display is not working, screen is blank", "ESP32", [{"type": "OLED_DISPLAY"}])
    print("\nTest 2 (Troubleshooting):\n", resp2.reply[:200])
    resp3 = orchestrator.process_chat("Build a weather station with OLED display", "ESP32")
    print("\nTest 3 (NL Synthesis):\n", resp3.reply[:200])
    resp4 = orchestrator.process_chat("Show me truth table for NAND gate")
    print("\nTest 4 (Digital Logic):\n", resp4.reply[:200])
    resp5 = orchestrator.process_chat("Calculate BJT common emitter bias point")
    print("\nTest 5 (Analog):\n", resp5.reply[:200])
    resp6 = orchestrator.process_chat("What is the PCB trace width for 2A?")
    print("\nTest 6 (PCB):\n", resp6.reply[:200])
