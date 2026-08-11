"""
VoltForge Custom Tokenizer - Implemented from scratch.
Byte-Pair Encoding (BPE) tokenizer tailored for electronics domain, C/C++ firmware, JSON contracts, and canvas netlists.
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple


SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[SYS]",
    "[USER]",
    "[ASSISTANT]",
    "[CODE]",
    "[CANVAS]",
    "[WIRING]",
    "[JSON]",
    "[UNKNOWN_SEARCH]",
    "[THINK]",
    "[/THINK]",
    "[FORMULA]",
    "[CIRCUIT]",
    "[SAFETY]",
]

# Domain-specific seed tokens to pre-populate vocabulary before BPE training.
# These ensure common electronics subwords get their own token IDs.
DOMAIN_SEED_TOKENS = [
    "analogRead", "analogWrite", "digitalWrite", "digitalRead",
    "pinMode", "INPUT", "OUTPUT", "INPUT_PULLUP",
    "Serial", "begin", "println", "print",
    "millis", "delay", "micros", "attachInterrupt",
    "Wire", "SPI", "I2C", "UART", "MOSI", "MISO", "SCK", "SDA", "SCL",
    "ATmega328P", "ATmega2560", "ATmega32U4", "ESP32", "RP2040", "STM32",
    "Arduino", "Uno", "Nano", "Mega", "ESP8266", "Pico", "Teensy",
    "LED", "resistor", "capacitor", "inductor", "transistor", "MOSFET",
    "diode", "relay", "servo", "motor", "stepper", "sensor", "buzzer",
    "IRLZ44N", "1N4007", "LM7805", "NE555", "LM358", "TL431",
    "SSD1306", "HD44780", "MAX7219", "ULN2003", "L298N", "A4988",
    "DHT11", "DHT22", "BMP280", "MPU6050", "HC-SR04", "DS18B20",
    "NeoPixel", "WS2812B", "ADS1115", "MCP3008", "PCF8574",
    "voltage", "current", "resistance", "capacitance", "inductance",
    "ohm", "farad", "henry", "ampere", "watt", "hertz",
    "GND", "VCC", "3.3V", "5V", "GPIO", "ADC", "DAC", "PWM",
    "HIGH", "LOW", "void", "setup", "loop", "int", "float", "const",
    "include", "define", "ifdef", "endif", "return", "while", "for",
    "if", "else", "switch", "case", "break", "unsigned", "long",
    "byte", "boolean", "char", "struct", "enum", "class",
]


class VoltForgeTokenizer:
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.special_tokens = list(SPECIAL_TOKENS)
        self.special_token_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_special_token = {idx: token for idx, token in enumerate(self.special_tokens)}
        
        self.vocab: Dict[str, int] = {}
        self.id_to_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        self.vocab = dict(self.special_token_to_id)
        # Base ASCII & common byte characters (indices starting after special tokens)
        next_id = len(self.vocab)
        for i in range(256):
            char = chr(i)
            if char not in self.vocab:
                self.vocab[char] = next_id
                next_id += 1
        # Pre-populate domain-specific seed tokens
        for seed in DOMAIN_SEED_TOKENS:
            if seed not in self.vocab:
                self.vocab[seed] = next_id
                next_id += 1
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}

    def _get_stats(self, words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        pairs: Dict[Tuple[str, str], int] = {}
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        new_words: Dict[Tuple[str, ...], int] = {}
        bigram = pair
        replacement = "".join(pair)
        for word, freq in words.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words[tuple(new_word)] = freq
        return new_words

    def _merge_vocab_batch(self, pairs: List[Tuple[str, str]], words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        pair_map = {pair: "".join(pair) for pair in pairs}
        new_words: Dict[Tuple[str, ...], int] = {}
        for word, freq in words.items():
            new_word = []
            i = 0
            w_len = len(word)
            while i < w_len:
                if i < w_len - 1:
                    bigram = (word[i], word[i + 1])
                    if bigram in pair_map:
                        new_word.append(pair_map[bigram])
                        i += 2
                        continue
                new_word.append(word[i])
                i += 1
            new_words[tuple(new_word)] = freq
        return new_words

    def train(self, texts: List[str], target_vocab_size: Optional[int] = None) -> None:
        if target_vocab_size:
            self.vocab_size = target_vocab_size

        self._init_base_vocab()
        word_freqs: Dict[Tuple[str, ...], int] = {}

        for text in texts:
            tokens = re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)
            for token in tokens:
                if not token:
                    continue
                char_tuple = tuple(list(token))
                word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1

        self.merges = []
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            
            sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
            batch_limit = min(64, self.vocab_size - len(self.vocab))
            
            pairs_to_merge = []
            used_subwords = set()
            for pair, freq in sorted_pairs:
                if freq < 2:
                    break
                if pair[0] in used_subwords or pair[1] in used_subwords:
                    continue
                pairs_to_merge.append(pair)
                used_subwords.add(pair[0])
                used_subwords.add(pair[1])
                used_subwords.add("".join(pair))
                if len(pairs_to_merge) >= batch_limit:
                    break

            if not pairs_to_merge:
                break

            word_freqs = self._merge_vocab_batch(pairs_to_merge, word_freqs)
            for pair in pairs_to_merge:
                new_token = "".join(pair)
                self.merges.append(pair)
                if new_token not in self.vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_token] = new_id
                    self.id_to_vocab[new_id] = new_token

    def _tokenize_word(self, word: str) -> List[str]:
        if not hasattr(self, "word_cache"):
            self.word_cache = {}
        if word in self.word_cache:
            return self.word_cache[word]

        if not hasattr(self, "ranks"):
            self.ranks = {tuple(pair): i for i, pair in enumerate(self.merges)}

        tokens = list(word)
        while len(tokens) >= 2:
            min_rank = float("inf")
            best_i = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.ranks.get(pair, None)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_i = i

            if best_i == -1:
                break

            pair = (tokens[best_i], tokens[best_i + 1])
            tokens = tokens[:best_i] + ["".join(pair)] + tokens[best_i + 2:]

        self.word_cache[word] = tokens
        return tokens

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.special_token_to_id["[BOS]"])

        # Check for special tokens in text
        special_pattern = "|".join(re.escape(st) for st in self.special_tokens)
        segments = re.split(f"({special_pattern})", text)

        for segment in segments:
            if not segment:
                continue
            if segment in self.special_token_to_id:
                ids.append(self.special_token_to_id[segment])
                continue

            words = re.findall(r"\w+|[^\w\s]|\s+", segment, re.UNICODE)
            for word in words:
                subwords = self._tokenize_word(word)
                for sw in subwords:
                    token_id = self.vocab.get(sw, self.special_token_to_id["[UNK]"])
                    ids.append(token_id)

        if add_eos:
            ids.append(self.special_token_to_id["[EOS]"])
        return ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        tokens: List[str] = []
        for tid in token_ids:
            if tid in self.id_to_special_token:
                if not skip_special:
                    tokens.append(self.id_to_special_token[tid])
            elif tid in self.id_to_vocab:
                tokens.append(self.id_to_vocab[tid])
            else:
                if not skip_special:
                    tokens.append("[UNK]")
        return "".join(tokens)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
        with open(os.path.join(directory, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(self.merges, f, indent=2, ensure_ascii=False)
        with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": len(self.vocab),
                "special_tokens": self.special_tokens
            }, f, indent=2)

    def load(self, directory: str) -> None:
        with open(os.path.join(directory, "vocab.json"), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(os.path.join(directory, "merges.json"), "r", encoding="utf-8") as f:
            self.merges = [tuple(m) for m in json.load(f)]
        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self.word_cache = {}
