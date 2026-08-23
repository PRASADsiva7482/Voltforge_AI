import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

BOARD_DICTIONARY = {
    "ARDUINO_UNO": ["arduino uno", "uno", "arduno", "arduno uno", "atmel 328p", "uno r3"],
    "ARDUINO_UNO_R4": ["arduino uno r4", "uno r4", "uno wifi"],
    "ARDUINO_MEGA": ["arduino mega", "mega", "mega 2560", "arduno mega"],
    "ARDUINO_NANO": ["arduino nano", "nano", "arduno nano"],
    "ESP32": ["esp32", "esp 32", "esp32 devkit", "esp-32", "esp32 wroom"],
    "ESP32_S3": ["esp32 s3", "esp32s3", "esp32-s3"],
    "ESP32_C3": ["esp32 c3", "esp32c3", "esp32-c3"],
    "RASPBERRY_PI_PICO": ["raspberry pi pico", "rpi pico", "pico", "rp2040", "rasbery pi pico"],
    "RASPBERRY_PI_PICO_W": ["pico w", "raspberry pi pico w", "rp2040 w"],
    "RASPBERRY_PI_PICO_2": ["pico 2", "raspberry pi pico 2", "rp2350"],
    "STM32_BLUE_PILL": ["stm32 blue pill", "blue pill", "bluepill", "stm32f103c8t6"],
    "STM32_BLACK_PILL": ["stm32 black pill", "black pill", "blackpill"],
    "TEENSY_4_0": ["teensy 4.0", "teensy 4", "teensy4"],
    "TEENSY_4_1": ["teensy 4.1", "teensy41"],
    "SEEED_XIAO_ESP32C3": ["xiao esp32c3", "seeed xiao esp32", "xiao esp32"],
}

COMPONENT_DICTIONARY = {
    "MPU6050": ["mpu6050", "mpu 6050", "mpu-6050", "gy-521", "accelerometer", "gyro", "gyroscope"],
    "BME280": ["bme280", "bme 280", "bme-280", "pressure sensor", "humidity sensor"],
    "DHT22": ["dht22", "dht11", "dht", "temperature sensor", "temp sensor"],
    "LDR_SENSOR": ["ldr", "photoresistor", "light sensor", "photo resistor", "photocell", "light dependent resistor"],
    "OLED_DISPLAY": ["oled", "oled display", "oled dispay", "ssd1306", "128x64 oled", "i2c oled"],
    "LCD_1602_I2C": ["lcd", "lcd 1602", "16x2 lcd", "i2c lcd", "liquid crystal"],
    "RELAY_MODULE": ["relay", "relay module", "rely", "realy", "5v relay", "1-channel relay"],
    "DC_MOTOR": ["dc motor", "motor", "dc-motor", "electric motor"],
    "SERVO_MOTOR": ["servo", "servo motor", "sg90", "mg996r", "micro servo"],
    "LED": ["led", "ledd", "light emitting diode", "indicator led"],
    "RESISTOR": ["resistor", "resisiter", "resistor 220", "pullup resistor"],
    "CAPACITOR": ["capacitor", "capasitor", "cap", "decoupling capacitor"],
    "ROTARY_ENCODER": ["rotary encoder", "encoder", "ky-040", "quadrature encoder"],
    "NEO6M_GPS": ["gps", "neo-6m", "neo6m", "gps module"],
    "MAX7219": ["max7219", "dot matrix", "8x8 matrix", "led matrix"],
    "BUTTON": ["button", "push button", "pushbutton", "switch", "tactile switch"],
    "LOGIC_LEVEL_CONVERTER": ["level shifter", "logic level converter", "level converter", "5v to 3.3v shifter"],
}




COMMON_STOPWORDS = {
    "build", "create", "design", "make", "with", "from", "for", "the", "and", "station",
    "system", "circuit", "project", "test", "what", "how", "why", "can", "tell", "show",
    "code", "file", "pin", "wire", "read", "check", "issue", "safe", "look", "like", "using",
    "please", "give", "help", "need", "want", "have", "some", "more", "does", "will", "this"
}


class FuzzyResolver:
    """Resilient Levenshtein distance and difflib fuzzy matcher for hardware terms and typos."""

    @classmethod
    def resolve_board(cls, query: str) -> Optional[str]:
        q_lower = query.lower().strip()
        if q_lower in COMMON_STOPWORDS or len(q_lower) < 3:
            return None

        # 1. Exact phrase or whole-word match in dictionary
        for board_key, aliases in BOARD_DICTIONARY.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, q_lower):
                    return board_key

        # 2. Strict fuzzy match on whole word only (cutoff >= 0.82)
        all_aliases = []
        alias_to_key = {}
        for board_key, aliases in BOARD_DICTIONARY.items():
            for alias in aliases:
                all_aliases.append(alias)
                alias_to_key[alias] = board_key

        matches = difflib.get_close_matches(q_lower, all_aliases, n=1, cutoff=0.82)
        if matches:
            return alias_to_key[matches[0]]

        return None

    @classmethod
    def resolve_component(cls, query: str) -> Optional[str]:
        q_lower = query.lower().strip()
        if q_lower in COMMON_STOPWORDS or len(q_lower) < 3:
            return None

        # 1. Exact phrase or whole-word match in dictionary
        for comp_key, aliases in COMPONENT_DICTIONARY.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, q_lower):
                    return comp_key

        # 2. Strict fuzzy match on whole word only (cutoff >= 0.82)
        all_aliases = []
        alias_to_key = {}
        for comp_key, aliases in COMPONENT_DICTIONARY.items():
            for alias in aliases:
                all_aliases.append(alias)
                alias_to_key[alias] = comp_key

        matches = difflib.get_close_matches(q_lower, all_aliases, n=1, cutoff=0.82)
        if matches:
            return alias_to_key[matches[0]]

        return None

    @classmethod
    def normalize_text(cls, text: str) -> Dict[str, Any]:
        """Cleans typos, extracts matched board & components, and returns structured annotations."""
        resolved_boards: List[str] = []
        resolved_components: List[str] = []
        t_lower = text.lower().strip()

        # Step 1: Check full sentence for exact phrase matches
        for board_key, aliases in BOARD_DICTIONARY.items():
            for alias in aliases:
                if re.search(r"\b" + re.escape(alias) + r"\b", t_lower):
                    if board_key not in resolved_boards:
                        resolved_boards.append(board_key)

        for comp_key, aliases in COMPONENT_DICTIONARY.items():
            for alias in aliases:
                if re.search(r"\b" + re.escape(alias) + r"\b", t_lower):
                    if comp_key not in resolved_components:
                        resolved_components.append(comp_key)

        # Step 2: Check individual words for typo corrections (excluding stopwords)
        words = re.findall(r"[a-zA-Z0-9_\-]+", text)
        for word in words:
            w_lower = word.lower()
            if w_lower in COMMON_STOPWORDS or len(w_lower) < 4:
                continue

            b = cls.resolve_board(w_lower)
            if b and b not in resolved_boards:
                resolved_boards.append(b)

            c = cls.resolve_component(w_lower)
            if c and c not in resolved_components:
                resolved_components.append(c)

        return {
            "originalText": text,
            "resolvedBoards": resolved_boards,
            "resolvedComponents": resolved_components,
            "isTypoCorrected": bool(resolved_boards or resolved_components),
        }



if __name__ == "__main__":
    test_queries = ["arduno with mpu 6050", "esp 32 with oled dispay", "rasbery pi pico with rely"]
    for t in test_queries:
        res = FuzzyResolver.normalize_text(t)
        print(f"Query: '{t}' -> Resolved Boards: {res['resolvedBoards']}, Components: {res['resolvedComponents']}")
