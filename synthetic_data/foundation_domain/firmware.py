"""Six independently authored reference programs, each with one compiler defect.

Compiler acceptance proves syntax/linking for the exact target. It is not a
hardware, timing, electrical compatibility or runtime correctness receipt.
"""
from pathlib import Path
from data_governance.splitting.signatures import canonical, sha
from synthetic_data.verifiers import verify_board, toolchain_identity

AI = Path(__file__).resolve().parents[2]
BOARDS = {"ARDUINO_UNO": "arduino:avr:uno", "ARDUINO_MEGA": "arduino:avr:mega:cpu=atmega2560"}

# program, exact single call to corrupt, explanation, project assumptions
PROGRAMS = {
    "bounded-command-parser": ('''#include <Arduino.h>
char command[16];
uint8_t used = 0;
bool discarded = false;
void acceptCommand(char ch) {
  if (ch == '\\r') return;
  if (ch == '\\n') {
    if (!discarded) { command[used] = '\\0'; Serial.println(command); }
    used = 0; discarded = false; return;
  }
  if (discarded) return;
  if (used < sizeof(command) - 1) command[used++] = ch;
  else discarded = true;
}
void setup() { Serial.begin(9600); }
void loop() {
  if (Serial.available() > 0) acceptCommand((char)Serial.read());
}
''', "acceptCommand((char)Serial.read())", "acceptCommandMissing((char)Serial.read())", "The call names a function that is not declared. Restore acceptCommand. The fixed parser reserves one terminator byte, discards an overlong line, and resets on newline; it never uses an unbounded String.", "Serial command echo; 15 payload characters maximum; CR ignored; LF terminates a command."),
    "interrupt-counter-snapshot": ('''#include <Arduino.h>
volatile uint32_t pulseCount = 0;
uint32_t lastReport = 0;
void countPulse() { ++pulseCount; }
uint32_t takeSnapshot() {
  noInterrupts();
  uint32_t copy = pulseCount;
  interrupts();
  return copy;
}
void setup() {
  Serial.begin(9600);
  pinMode(2, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(2), countPulse, RISING);
}
void loop() {
  uint32_t now = millis();
  if ((uint32_t)(now - lastReport) >= 1000UL) {
    lastReport = now;
    Serial.println(takeSnapshot());
  }
}
''', "Serial.println(takeSnapshot())", "Serial.println(takeSnapshotMissing())", "The print call uses an undeclared snapshot function. Restore takeSnapshot. The volatile multi-byte counter is copied with interrupts disabled from the normal loop, then interrupts are enabled again. The count wraps modulo 2^32; this is not a frequency calibration.", "Digital pin 2 interrupt input; an external compatible pulse source and common ground are assumed. No pulse-source electrical rating or maximum frequency is certified."),
    "adc-window-average": ('''#include <Arduino.h>
uint16_t samples[8] = {0};
uint32_t total = 0;
uint8_t cursor = 0, filled = 0;
uint32_t lastSample = 0;
uint16_t addSample(uint16_t sample) {
  total -= samples[cursor]; samples[cursor] = sample; total += sample;
  cursor = (cursor + 1) & 7;
  if (filled < 8) ++filled;
  return (uint16_t)(total / filled);
}
void setup() { Serial.begin(9600); }
void loop() {
  uint32_t now = millis();
  if ((uint32_t)(now - lastSample) >= 10UL) {
    lastSample = now;
    Serial.println(addSample((uint16_t)analogRead(A0)));
  }
}
''', "addSample((uint16_t)analogRead(A0))", "addSampleMissing((uint16_t)analogRead(A0))", "The moving-average call has an undeclared name. Restore addSample. Subtract the replaced sample before adding the new value. During startup divide by the number of collected samples; integer division truncates. Sampling timing is cooperative, not a guaranteed fixed-rate ADC clock.", "A0 provides raw ADC codes; no conversion to volts or sensor identity is assumed. Eight uint16 samples and a uint32 sum are used."),
    "cooperative-debouncer": ('''#include <Arduino.h>
uint8_t observed = HIGH, stable = HIGH;
uint32_t changedAt = 0;
void updateSwitch(uint8_t value, uint32_t now) {
  if (value != observed) { observed = value; changedAt = now; }
  if (stable != observed && (uint32_t)(now - changedAt) >= 25UL) {
    stable = observed; Serial.println(stable == LOW ? "pressed" : "released");
  }
}
void setup() { Serial.begin(9600); pinMode(4, INPUT_PULLUP); }
void loop() { updateSwitch(digitalRead(4), millis()); }
''', "updateSwitch(digitalRead(4), millis())", "updateSwitchMissing(digitalRead(4), millis())", "The loop calls a nonexistent debouncer. Restore updateSwitch. A raw transition restarts the quiet interval; publish only after 25 ms without a further transition. Internal pull-up makes a switch to ground active-low. This is a reference debounce interval, not a switch specification.", "Momentary switch between digital pin 4 and GND, using INPUT_PULLUP; serial reports stable transitions."),
    "serial-queue-drain": ('''#include <Arduino.h>
uint8_t queueBytes[16];
uint8_t head = 0, tail = 0;
bool enqueueByte(uint8_t value) {
  uint8_t next = (head + 1) & 15;
  if (next == tail) return false;
  queueBytes[head] = value; head = next; return true;
}
void setup() { Serial.begin(9600); }
void loop() {
  if (Serial.available() > 0) {
    uint8_t incoming = (uint8_t)Serial.read();
    if (!enqueueByte(incoming)) { /* Drop newest when full. */ }
  }
  if (tail != head && Serial.availableForWrite() > 0) {
    Serial.write(queueBytes[tail]); tail = (tail + 1) & 15;
  }
}
''', "enqueueByte(incoming)", "enqueueByteMissing(incoming)", "The producer calls an undeclared queue function. Restore enqueueByte. The reserved slot distinguishes full from empty, giving 15 usable bytes. The fixed code drops the newest byte on overflow and drains only when serial output has capacity; both operations run in the loop.", "Single-threaded 16-slot serial queue; no ISR producer, dynamic allocation or lossless throughput claim."),
    "wire-register-transaction": ('''#include <Arduino.h>
#include <Wire.h>
const uint8_t deviceAddress = 0x2A, registerAddress = 0x03;
uint32_t lastPoll = 0;
bool readRegister(uint8_t &value) {
  Wire.beginTransmission(deviceAddress);
  Wire.write(registerAddress);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(deviceAddress, (uint8_t)1, (uint8_t)true) != 1) return false;
  if (Wire.available() < 1) return false;
  value = (uint8_t)Wire.read(); return true;
}
void setup() { Serial.begin(9600); Wire.begin(); }
void loop() {
  uint32_t now = millis();
  if ((uint32_t)(now - lastPoll) >= 1000UL) {
    lastPoll = now;
    uint8_t value;
    if (readRegister(value)) Serial.println(value);
    else Serial.println("transaction failed");
  }
}
''', "readRegister(value)", "readRegisterMissing(value)", "The loop uses an undeclared register-reader name. Restore readRegister. Check the address phase status, received count and available byte before reading. The no-STOP address phase requests a repeated START. Wire calls may block; this example does not establish bus timeout or recovery behavior.", "Hypothetical I2C device supplied by the problem at 7-bit address 0x2A and register 0x03, returning one byte. No real part is claimed. Board-specific SDA/SCL and suitable pull-ups must be selected for an actual device."),
}


def exact_board(board):
    if board not in BOARDS: raise ValueError("Unsupported exact recipe target; no board alias substitution")
    return verify_board(board) | {"fqbn": BOARDS[board]}


def cases(family, board):
    source, call, broken_call, _, _ = PROGRAMS[family]
    if source.count(call) != 1: raise ValueError("Compiler mutation is not singular")
    target = exact_board(board)
    return [{"caseId": f"vf-g2-domain-v1:{family}:{board}:{state}", "fqbn": target["fqbn"],
             "toolchainId": "arduino-avr-1.8.6", "source": text, "expectedSuccess": success}
            for state, text, success in (("broken", source.replace(call, broken_call), False), ("fixed", source, True))]


def build_environment():
    """Bind every local AVR core/library byte, not just a claimed version string."""
    core = AI / ".toolchains/arduino-data/packages/arduino/hardware/avr/1.8.6"
    if not core.is_dir(): raise ValueError("Pinned AVR core is missing")
    files = [{"path": path.relative_to(core).as_posix(), "sha256": sha(path.read_bytes())} for path in sorted(core.rglob("*")) if path.is_file()]
    props = (core / "libraries/Wire/library.properties").read_text(encoding="utf-8")
    version = next(line.split("=", 1)[1] for line in props.splitlines() if line.startswith("version="))
    return {"toolchain": toolchain_identity("arduino-avr-1.8.6", require_files=True), "coreTreeSha256": sha(canonical(files)),
            "coreFiles": files, "libraries": [{"name": "Wire", "version": version, "source": "pinned-core-bundled", "externalLibraries": []}],
            "coreRelativePath": core.relative_to(AI).as_posix(), "targets": {board: exact_board(board) for board in BOARDS}}
