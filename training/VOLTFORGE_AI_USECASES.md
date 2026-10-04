# VoltForge AI - Comprehensive Electronics Domain & Use Case Specification

This document details all verified use cases, hardware targets, component specifications, circuit safety rules, and firmware synthesis patterns for training the **VoltForge AI** copilot model.

---

## 1. System Overview & Integration Flow

VoltForge is a browser-based Virtual Electronics Simulator. The AI serves as an active electronics copilot inside `Voltforge_UI` (React/Vite) and communicates via `Voltforge_BL` (Spring Boot) and `Voltforge_AI` (FastAPI).

### Runtime Interaction Cycle
1. **User Canvas State**: The user places microcontroller boards, ICs, sensors, displays, and wiring on the canvas.
2. **Context Compilation**: The frontend serializes:
   - `boardType`: e.g. `ARDUINO_UNO`, `ESP32`, `RASPBERRY_PI_PICO`, `STM32_BLUE_PILL`
   - `components`: Node list with types, IDs, properties (e.g. resistance `220`, capacitance `10uF`)
   - `wires`: Connections (`fromComponent:fromPin` -> `toComponent:toPin`)
   - `netlist`: Electrical nodes and pin mappings
   - `code`: Active C++ sketch in the Monaco editor
   - `simulationState`: Running status, pin voltages, solver convergence
3. **AI Task Execution**: The model receives the user's natural language request alongside the canvas context and provides:
   - Clean, direct answers without unnecessary hallucinated component suggestions
   - Compilable C++ firmware matching the user's exact canvas pins
   - Verified pinout routing proposals
   - Circuit safety and Design Rule Check (DRC) verdicts

---

## 2. Core AI Use Cases

### Use Case 1: Circuit Safety & Design Rule Check (DRC)
- **Goal**: Detect dangerous electrical conditions before or during simulation.
- **Triggers**: *"Is this circuit valid?"*, *"Check for short circuits"*, *"Is my wiring safe?"*, *"Validate circuit"*.
- **Rules Evaluated**:
  1. **Short Circuit Detection**: Power rails (VCC/5V/3.3V) directly connected to Ground (GND) with negligible resistance.
  2. **LED Current Limiting**: LEDs connected across power rails or GPIO without a series resistor (requires 220Ω–330Ω to limit current to ~15mA).
  3. **Logic Level Voltage Incompatibility**:
     - 3.3V Boards (`ESP32`, `RASPBERRY_PI_PICO`, `STM32_BLUE_PILL`, `TEENSY_4_0`, `SEEED_XIAO`) cannot tolerate 5V signals on GPIO.
     - 5V sensors (e.g. HC-SR04 ECHO pin, 5V UART) must be level-shifted (using a bidirectional converter or 10kΩ/20kΩ voltage divider) before entering 3.3V pins.
  4. **Inductive Kickback Protection**:
     - Inductive loads (`DC_MOTOR`, `RELAY_MODULE`, `SOLENOID`, `STEPPER_MOTOR`) must have a flyback diode (e.g. 1N4007) in anti-parallel across the coil to suppress reverse EMF spikes.
     - Never connect motors or relay coils directly to MCU GPIO pins; require a driver (MOSFET, transistor, ULN2003, or L298N).
  5. **Floating Inputs**:
     - High-impedance input pins (pushbuttons, digital sensors) require pull-up or pull-down resistors (4.7kΩ–10kΩ), or `pinMode(pin, INPUT_PULLUP)` in code.
  6. **I2C Bus Termination**:
     - I2C lines (SDA, SCL) require 4.7kΩ pull-up resistors to VCC if not integrated into the sensor breakout board.
  7. **Common Ground**:
     - Microcontroller GND and external power supply GND must be tied together to establish a shared reference.

### Use Case 2: Intelligent Pin Routing & Wiring Suggestions
- **Goal**: Provide correct, board-specific pin connections for newly placed components.
- **Triggers**: *"How do I connect the OLED display?"*, *"Wire DHT22 to ESP32"*, *"What pins should I connect to?"*.
- **Standard Protocol Pins**:
  - **I2C (SDA / SCL)**:
    - `ARDUINO_UNO` / `ARDUINO_NANO`: SDA = `A4`, SCL = `A5`
    - `ARDUINO_MEGA`: SDA = `D20`, SCL = `D21`
    - `ESP32`: SDA = `D21`, SCL = `D22`
    - `ESP8266`: SDA = `D2 (GPIO4)`, SCL = `D1 (GPIO5)`
    - `RASPBERRY_PI_PICO`: SDA = `GP4 (pin 6)`, SCL = `GP5 (pin 7)`
    - `STM32_BLUE_PILL`: SDA = `PB7`, SCL = `PB6`
  - **SPI (MOSI / MISO / SCK / CS)**:
    - `ARDUINO_UNO`: MOSI = `D11`, MISO = `D12`, SCK = `D13`, CS = `D10`
    - `ESP32`: MOSI = `D23`, MISO = `D19`, SCK = `D18`, CS = `D5`
    - `RASPBERRY_PI_PICO`: MOSI = `GP19`, MISO = `GP16`, SCK = `GP18`, CS = `GP17`
  - **UART (TX / RX)**:
    - `ARDUINO_UNO`: TX = `D1`, RX = `D0`
    - `ESP32`: TX = `GPIO1`, RX = `GPIO3` (UART0) or TX = `GPIO17`, RX = `GPIO16` (UART2)
  - **ADC (Analog Inputs)**:
    - `ARDUINO_UNO`: `A0`–`A5` (10-bit, 0–5V)
    - `ESP32`: `GPIO32`–`GPIO39` (ADC1, recommended with Wi-Fi). Avoid ADC2 (`GPIO0, 2, 4, 12-15, 25-27`) when Wi-Fi is active.
    - `RASPBERRY_PI_PICO`: `GP26 (A0)`, `GP27 (A1)`, `GP28 (A2)` (12-bit, 0–3.3V)
  - **Strapping & Restricted Pins**:
    - `ESP32`: Avoid using strapping pins (`GPIO 0, 2, 5, 12, 15`) for input switches or heavy pull-ups, as they alter boot modes.
    - `ESP32`: `GPIO 34, 35, 36, 39` are input-only (no internal pull-ups/pull-downs, cannot be outputs).

### Use Case 3: Schematic-to-Firmware Code Synthesis
- **Goal**: Generate clean, compilable, complete C++ sketches matching the exact canvas components and wired pins.
- **Triggers**: *"Generate Arduino code for my placed circuit"*, *"Write code to read temperature and show on LCD"*, *"Blink sketch"*.
- **Rules**:
  - Always output code inside ```cpp ... ``` blocks.
  - Pin numbers in code must match the wires connected on canvas (e.g. if LED is on D7, use `const int LED_PIN = 7;`).
  - Include necessary headers (`#include <Arduino.h>`, `#include <Wire.h>`, `#include <SPI.h>`, `#include <Adafruit_SSD1306.h>`).
  - Configure `setup()` with correct `pinMode()` settings and initialize `Serial.begin(115200);`.
  - Provide non-blocking logic (`millis()`) where appropriate instead of excessive `delay()`.

### Use Case 4: Compiler Error Diagnostics & Firmware Repair
- **Goal**: Identify compilation errors from the simulation or Arduino toolchain and generate fixed code.
- **Triggers**: Compiler errors pasted by user (e.g. `'LED_PIN' was not declared in this scope`, `expected ';' before '}' token`, `no matching function for call to 'DHT::readTemperature()'`).
- **Rules**:
  - Identify the exact missing token, syntax issue, or mismatched library call.
  - Provide the corrected, complete code block.

### Use Case 5: Electrical Calculations & Component Sizing
- **Goal**: Perform Ohm's law, power dissipation, voltage divider, and timing calculations.
- **Formulas**:
  - **Ohm's Law**: $V = I \times R$, $I = \frac{V}{R}$, $R = \frac{V}{I}$
  - **Power Dissipation**: $P = V \times I = I^2 \times R = \frac{V^2}{R}$
  - **LED Resistor**: $R = \frac{V_{supply} - V_{forward}}{I_{target}}$ (e.g., Red LED on 5V at 15mA: $\frac{5.0 - 2.0}{0.015} = 200\Omega \rightarrow 220\Omega$ standard).
  - **Voltage Divider**: $V_{out} = V_{in} \times \frac{R_2}{R_1 + R_2}$ (5V to 3.3V divider: $R_1 = 10k\Omega, R_2 = 20k\Omega \rightarrow 3.33V$).
  - **NE555 Astable Frequency**: $f = \frac{1.44}{(R_1 + 2R_2)C}$

### Use Case 6: Simulation Troubleshooting
- **Goal**: Diagnose why a circuit is not behaving as expected in the simulator.
- **Triggers**: *"Why is the LED not glowing?"*, *"Motor is not spinning"*, *"Display is blank"*.
- **Diagnostics**:
  - Check whether simulation is running (`isSimulating`).
  - Check component polarity (LED anode vs. cathode, electrolytic capacitor polarity).
  - Check missing power (VCC not wired or switch open).
  - Check missing common ground.
  - Check I2C address conflict or incorrect Wire speed.

---

## 3. Supported Hardware Catalog

### Microcontroller Boards (30 Verified Platforms)
| Board Type | Family | Core / Processor | Logic Voltage | Clock | Compiler |
|---|---|---|---|---|---|
| `ARDUINO_UNO` | Arduino | ATmega328P | 5.0 V | 16 MHz | AVR8js / Arduino-core |
| `ARDUINO_UNO_R4` | Arduino | Renesas RA4M1 | 5.0 V | 48 MHz | Arduino-core |
| `ARDUINO_NANO` | Arduino | ATmega328P | 5.0 V | 16 MHz | AVR8js / Arduino-core |
| `ARDUINO_NANO_EVERY` | Arduino | ATmega4809 | 5.0 V | 20 MHz | Arduino-core |
| `ARDUINO_NANO_33_IOT`| Arduino | SAMD21 + Wi-Fi | 3.3 V | 48 MHz | Arduino-core |
| `ARDUINO_MEGA` | Arduino | ATmega2560 | 5.0 V | 16 MHz | AVR8js / Arduino-core |
| `ARDUINO_LEONARDO` | Arduino | ATmega32U4 | 5.0 V | 16 MHz | Arduino-core |
| `ARDUINO_MICRO` | Arduino | ATmega32U4 | 5.0 V | 16 MHz | Arduino-core |
| `ARDUINO_DUE` | Arduino | SAM3X8E Cortex-M3 | 3.3 V | 84 MHz | Arduino-core |
| `ARDUINO_GIGA_R1` | Arduino | STM32H747XI Dual | 3.3 V | 480 MHz | Arduino-core |
| `ARDUINO_PORTENTA_H7`| Arduino | STM32H747 Dual | 3.3 V | 480 MHz | Arduino-core |
| `ESP8266` | ESP8266 | ESP-12E / NodeMCU | 3.3 V | 80 MHz | Arduino-core |
| `ESP8266_WEMOS_D1` | ESP8266 | ESP8266EX | 3.3 V | 80 MHz | Arduino-core |
| `ESP32` | ESP32 | ESP32-D0WDQ6 Dual | 3.3 V | 240 MHz | Arduino-core |
| `ESP32_WROOM` | ESP32 | ESP32-WROOM-32 | 3.3 V | 240 MHz | Arduino-core |
| `ESP32_WROVER` | ESP32 | ESP32 with PSRAM | 3.3 V | 240 MHz | Arduino-core |
| `ESP32_S2` | ESP32 | ESP32-S2 Single | 3.3 V | 240 MHz | Arduino-core |
| `ESP32_S3` | ESP32 | ESP32-S3 Dual AI | 3.3 V | 240 MHz | Arduino-core |
| `ESP32_C3` | ESP32 | RISC-V Single Core| 3.3 V | 160 MHz | Arduino-core |
| `ESP32_C6` | ESP32 | RISC-V Wi-Fi 6 | 3.3 V | 160 MHz | Arduino-core |
| `RASPBERRY_PI_PICO` | RP2040 | Dual Cortex-M0+ | 3.3 V | 133 MHz | Arduino-core / Pico SDK |
| `RASPBERRY_PI_PICO_W`| RP2040 | Dual Cortex-M0+ | 3.3 V | 133 MHz | Arduino-core / Pico SDK |
| `RASPBERRY_PI_PICO_2`| RP2350 | Dual Cortex-M33 | 3.3 V | 150 MHz | Arduino-core / Pico SDK |
| `STM32_BLUE_PILL` | STM32 | STM32F103C8T6 | 3.3 V | 72 MHz | STM32duino |
| `STM32_BLACK_PILL` | STM32 | STM32F411CEU6 | 3.3 V | 100 MHz | STM32duino |
| `TEENSY_4_0` | Teensy | i.MX RT1062 M7 | 3.3 V | 600 MHz | Teensyduino |
| `TEENSY_4_1` | Teensy | i.MX RT1062 M7 | 3.3 V | 600 MHz | Teensyduino |
| `SEEED_XIAO_SAMD21` | Seeed | SAMD21 Cortex-M0+ | 3.3 V | 48 MHz | Arduino-core |
| `SEEED_XIAO_ESP32C3`| Seeed | ESP32-C3 RISC-V | 3.3 V | 160 MHz | Arduino-core |
| `ATMEL_AVR_ATMEGA328P`| AVR DIP | ATmega328P 28-pin| 5.0 V | 16 MHz | AVR8js |

---

## 4. Component Catalog & Electrical Characteristics

### Passive Components
- `RESISTOR`: Current limiting, pull-up, pull-down, voltage divider.
- `CAPACITOR` / `CERAMIC_CAPACITOR`: Decoupling (100nF), noise filtering.
- `ELECTROLYTIC_CAPACITOR`: Bulk power rail stabilization (10μF–470μF, polarized).
- `DIODE` (1N4007): Rectification, reverse polarity protection, flyback diode.
- `ZENER_DIODE`: Voltage clamping, shunt voltage reference (e.g. 5.1V, 3.3V).
- `POTENTIOMETER`: Variable voltage divider, analog tuning wiper.

### Actuators & Motors
- `STANDARD_LED` / `LED_RED` / `LED_GREEN` / `LED_BLUE`: $V_f \approx 2.0V$ (Red/Green) or $3.2V$ (Blue/White), $I_{max} \approx 20mA$.
- `RGB_LED`: Common cathode or common anode, 3 distinct current-limiting resistors.
- `SERVO_MOTOR` (SG90): 5V power, PWM control signal (1ms–2ms pulse width for 0°–180°), peak current up to 500mA (needs stable power).
- `DC_MOTOR`: Inductive load, requires driver transistor/MOSFET + 1N4007 flyback diode.
- `STEPPER_MOTOR` (28BYJ-48): 4-phase unipolar stepper, driven via `ULN2003` Darlington array.
- `RELAY_SINGLE` / `RELAY_MODULE`: 5V coil, COM, NO, NC contacts, active-low input on modules.
- `BUZZER_ACTIVE` / `BUZZER_PASSIVE`: Active needs DC voltage; passive needs square wave frequency (`tone()`).

### Sensors & Input Devices
- `PUSH_BUTTON` / `SPST_SWITCH`: Momentary or toggle switch, requires pull-up or pull-down.
- `DHT11` / `DHT22`: Temperature & humidity, single-wire bus, 4.7kΩ pull-up required.
- `HC_SR04_ULTRASONIC`: TRIG (10μs trigger pulse), ECHO (5V pulse width output; needs level shift for 3.3V boards).
- `MPU6050`: 6-axis Gyroscope & Accelerometer, I2C address `0x68` (AD0 low) or `0x69` (AD0 high).
- `PIR_MOTION_SENSOR`: Passive infrared motion detection, digital OUT signal.
- `PHOTORESISTOR_LDR`: Light-dependent resistor, used in voltage divider with 10kΩ fixed resistor.

### Displays
- `OLED_SSD1306` (128x64 or 128x32): I2C interface, address `0x3C` (default) or `0x3D`.
- `LCD_1602_I2C`: 16x2 character display with PCF8574 I2C backpack, address `0x27` or `0x3F`.
- `SEVEN_SEGMENT`: Single or multi-digit LED displays, requires current limiting per segment.

### Integrated Circuits (ICs)
- `NE555_TIMER`: Astable multivibrator (square wave clock) or monostable pulse timer.
- `LM358_OPAMP`: Dual general-purpose operational amplifier (comparator, amplifier, active filter).
- `74HC595`: 8-bit serial-in, parallel-out shift register (pin expansion for LEDs/relays).
- `ULN2003`: 7-channel Darlington transistor array with internal clamp diodes for inductive loads.
- `LM7805`: 5V 1A linear voltage regulator.
