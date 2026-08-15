import json
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("voltforge-ai.web_search")

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasheet_cache.json")


class ComponentSpecExtractor:
    """Extracts electrical specs (voltage, interfaces, addresses, pinouts) from text snippets."""

    VOLTAGE_PATTERN = re.compile(
        r'(\b\d+(?:\.\d+)?\s*(?:V|volts|VDC)\b(?:\s*-\s*\d+(?:\.\d+)?\s*(?:V|volts|VDC)\b)?)',
        re.IGNORECASE,
    )
    I2C_ADDR_PATTERN = re.compile(r'\b(0x[0-9a-fA-F]{2})\b')
    BUS_INTERFACE_PATTERNS = {
        "I2C": re.compile(r'\b(I2C|I^2C|TWI|SMBus)\b', re.IGNORECASE),
        "SPI": re.compile(r'\b(SPI|Serial Peripheral Interface)\b', re.IGNORECASE),
        "UART": re.compile(r'\b(UART|USART|Serial|TX/RX)\b', re.IGNORECASE),
        "PWM": re.compile(r'\b(PWM|Pulse Width Modulation)\b', re.IGNORECASE),
        "ADC": re.compile(r'\b(ADC|Analog|Analog-to-Digital)\b', re.IGNORECASE),
        "CAN": re.compile(r'\b(CAN bus|Controller Area Network)\b', re.IGNORECASE),
    }

    @classmethod
    def extract_specs(cls, query: str, snippets: List[Dict[str, str]]) -> Dict[str, Any]:
        combined_text = " ".join([f"{s.get('title', '')} {s.get('snippet', '')}" for s in snippets])

        voltages = list(set(cls.VOLTAGE_PATTERN.findall(combined_text)))
        i2c_addresses = list(set(cls.I2C_ADDR_PATTERN.findall(combined_text)))

        interfaces = [
            name for name, pattern in cls.BUS_INTERFACE_PATTERNS.items() if pattern.search(combined_text)
        ]

        operating_voltage = None
        if voltages:
            # Pick standard voltage ranges if available
            preferred = [v for v in voltages if any(term in v for term in ["3.3", "5", "3.3V", "5V", "1.8"])]
            operating_voltage = preferred[0] if preferred else voltages[0]

        return {
            "query": query,
            "operatingVoltage": operating_voltage or "3.3V / 5V (Standard)",
            "voltagesFound": voltages[:5],
            "i2cAddresses": i2c_addresses[:4],
            "supportedInterfaces": interfaces,
            "summarySnippet": combined_text[:300] + "..." if len(combined_text) > 300 else combined_text,
        }


class WebSearchEngine:
    """Multi-source search engine with dynamic caching for electronic components."""

    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load datasheet cache: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save datasheet cache: {e}")

    def search_duckduckgo(self, query: str, limit: int = 4) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            encoded = urllib.parse.quote(f"{query} datasheet pinout wiring specs")
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            resp = requests.get(url, headers=headers, timeout=1.5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for element in soup.select(".result__body")[:limit]:
                    title_elem = element.select_one(".result__title .result__a")
                    snippet_elem = element.select_one(".result__snippet")
                    if title_elem and snippet_elem:
                        results.append({
                            "title": title_elem.get_text(strip=True),
                            "url": title_elem.get("href", ""),
                            "snippet": snippet_elem.get_text(" ", strip=True),
                            "source": "DuckDuckGo Web",
                        })
        except Exception as e:
            logger.warning(f"DuckDuckGo search error for '{query}': {e}")
        return results

    def search_wikipedia(self, query: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json"
            resp = requests.get(url, timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                search_items = data.get("query", {}).get("search", [])
                for item in search_items[:2]:
                    snippet_clean = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                    results.append({
                        "title": item.get("title", ""),
                        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(item.get('title', ''))}",
                        "snippet": snippet_clean,
                        "source": "Wikipedia API",
                    })
        except Exception as e:
            logger.warning(f"Wikipedia search error for '{query}': {e}")
        return results

    def get_component_info(self, component_name: str, force_refresh: bool = False) -> Dict[str, Any]:
        key = component_name.lower().strip()
        if not force_refresh and key in self.cache:
            logger.info(f"Returning cached specs for '{component_name}'")
            return self.cache[key]

        # Check local offline knowledge graph first
        try:
            from engine.knowledge_graph import ComponentKnowledgeGraph
            kg_info = ComponentKnowledgeGraph.get_component_info(component_name)
            if kg_info:
                return {
                    "component": component_name,
                    "searchResults": [{"title": kg_info.get("fullName", component_name), "snippet": f"{kg_info.get('fullName')}: {', '.join(kg_info.get('requiredExternalComponents', []))}", "source": "VoltForge Local Knowledge Graph", "url": "local://datasheet"}],

                    "specs": {
                        "query": component_name,
                        "operatingVoltage": f"{kg_info.get('operatingVoltage', {}).get('min', 3.3)}V - {kg_info.get('operatingVoltage', {}).get('max', 5.0)}V",
                        "supportedInterfaces": kg_info.get("interfaces", []),
                        "i2cAddresses": kg_info.get("i2cAddress", []),
                        "summarySnippet": f"{kg_info.get('fullName', '')} ({kg_info.get('category', '')}). Interfaces: {', '.join(kg_info.get('interfaces', []))}. Current: {kg_info.get('currentDraw_mA', 0)}mA.",
                    },
                    "citations": [{"title": kg_info.get("fullName", component_name), "url": "local://knowledge-base", "source": "VoltForge Offline Database"}],
                }
        except Exception as e:
            logger.warning(f"Local KG lookup error: {e}")

        logger.info(f"Searching web for component '{component_name}'...")
        web_results = self.search_duckduckgo(component_name)
        if not web_results:
            web_results = self.search_wikipedia(component_name)


        extracted = ComponentSpecExtractor.extract_specs(component_name, web_results)
        result = {
            "component": component_name,
            "searchResults": web_results,
            "specs": extracted,
            "citations": [
                {"title": r["title"], "url": r["url"], "source": r.get("source", "Web")}
                for r in web_results
            ],
        }

        if web_results:
            self.cache[key] = result
            self._save_cache()

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = WebSearchEngine()
    test_res = engine.get_component_info("MPU6050")
    print("Extracted Specs for MPU6050:", json.dumps(test_res["specs"], indent=2))
