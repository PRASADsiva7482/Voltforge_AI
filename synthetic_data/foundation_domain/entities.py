"""Exact entity evidence; generic component labels never acquire device ratings."""
from electronics_corpus import get_electronics_corpus
from data_governance.splitting.signatures import canonical, sha

CASES = [
    ("led-identity", "component", "LED", "For the component labelled LED, can an OLED module datasheet establish its forward current?", "No. A light-emitting diode is a different entity from an OLED display module. Select the LED manufacturer and exact part number; forward voltage and current remain unknown.", "unknown"),
    ("generic-relay-ratings", "component", "RELAY", "A project contains a generic relay module. What coil voltage and load current rating can be assigned?", "Both remain unknown for this generic module. The exact module and relay part, driver circuit, isolation and load category are needed before electrical ratings can be assigned.", "unknown"),
    ("generic-display-variant", "component", "OLED", "A bill of materials says only OLED. Does that establish which supply voltage and level shifting this module supports?", "No exact display module is identified. A controller family or generic display label does not establish breakout-board supply or logic compatibility; request the manufacturer and module revision.", "unknown"),
    ("false-board-identity", "board", "Arduino Uno ESP32 R3", "Can a board described as Arduino Uno ESP32 R3 inherit the Uno R3 compile target and AVR memory figures?", "That mixed board name does not identify a supported exact variant. Do not substitute a different board's MCU, memory, pin map or FQBN; obtain the manufacturer and exact board revision.", "unknown"),
]


def evidence(kind, query):
    corpus = get_electronics_corpus()
    found = corpus.lookup(kind, query)
    return {"status": found.status, "reasonCode": found.reasonCode, "query": query, "kind": kind,
            "records": [{"recordId": row["recordId"], "sha256": sha(canonical(row))} for row in found.records],
            "genericRatingsKnown": False}
