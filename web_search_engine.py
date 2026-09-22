"""Compatibility adapter over the VFAI-023 secure internet evidence service.

New code should import :mod:`internet_retrieval` directly. This class preserves
the historical datasheet endpoints and reasoning-engine call surface without
retaining the former unrestricted HTML scraper or disk cache.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from config import get_settings
from internet_retrieval import (
    InternetRetrievalQuery,
    InternetRetrievalService,
    citations_from_response,
    response_metadata,
    service_from_settings,
)


logger = logging.getLogger("voltforge-ai.web-search-compatibility")


class ComponentSpecExtractor:
    """Extract only bounded values that occur in accepted evidence snippets."""

    VOLTAGE_PATTERN = re.compile(
        r"(\b\d+(?:\.\d+)?\s*(?:V|volts|VDC)\b(?:\s*(?:-|to)\s*\d+(?:\.\d+)?\s*(?:V|volts|VDC)\b)?)",
        re.IGNORECASE,
    )
    I2C_ADDR_PATTERN = re.compile(r"\b(0x[0-9a-fA-F]{2})\b")
    BUS_INTERFACE_PATTERNS = {
        "I2C": re.compile(r"\b(I2C|I\^2C|TWI|SMBus)\b", re.IGNORECASE),
        "SPI": re.compile(r"\b(SPI|Serial Peripheral Interface)\b", re.IGNORECASE),
        "UART": re.compile(r"\b(UART|USART|Serial|TX/RX)\b", re.IGNORECASE),
        "PWM": re.compile(r"\b(PWM|Pulse Width Modulation)\b", re.IGNORECASE),
        "ADC": re.compile(r"\b(ADC|Analog-to-Digital)\b", re.IGNORECASE),
        "CAN": re.compile(r"\b(CAN bus|Controller Area Network)\b", re.IGNORECASE),
    }

    @classmethod
    def extract_specs(cls, query: str, snippets: List[Dict[str, str]]) -> Dict[str, Any]:
        combined_text = " ".join(
            f"{item.get('title', '')} {item.get('snippet', '')}" for item in snippets[:4]
        )
        voltages = sorted(set(cls.VOLTAGE_PATTERN.findall(combined_text)), key=str.casefold)[:5]
        addresses = sorted(set(cls.I2C_ADDR_PATTERN.findall(combined_text)), key=str.casefold)[:4]
        interfaces = [
            name
            for name, pattern in cls.BUS_INTERFACE_PATTERNS.items()
            if pattern.search(combined_text)
        ]
        preferred = [
            voltage
            for voltage in voltages
            if any(term in voltage.casefold() for term in ("1.8", "3.3", "5v", "5 v"))
        ]
        return {
            "query": query,
            "operatingVoltage": (preferred[0] if preferred else voltages[0]) if voltages else None,
            "voltagesFound": voltages,
            "i2cAddresses": addresses,
            "supportedInterfaces": interfaces,
            "summarySnippet": (
                f"{combined_text[:300]}..." if len(combined_text) > 300 else combined_text
            ),
        }


class WebSearchEngine:
    """Local-first compatibility facade with secure, opt-in internet fallback."""

    def __init__(
        self,
        cache_file: str | None = None,
        internet_enabled: Optional[bool] = None,
        timeout_seconds: Optional[int] = None,
        service: InternetRetrievalService | None = None,
    ) -> None:
        _ = cache_file  # Disk caching was intentionally removed by VFAI-023.
        settings = get_settings()
        if internet_enabled is not None or timeout_seconds is not None:
            settings = settings.__class__(
                **{
                    **settings.__dict__,
                    "internet_retrieval_enabled": (
                        settings.internet_retrieval_enabled
                        if internet_enabled is None
                        else internet_enabled
                    ),
                    "internet_retrieval_timeout_seconds": (
                        settings.internet_retrieval_timeout_seconds
                        if timeout_seconds is None
                        else timeout_seconds
                    ),
                }
            )
        self.service = service or service_from_settings(settings)
        self.internet_enabled = self.service.enabled
        self.timeout_seconds = self.service.timeout_seconds

    def health(self) -> Dict[str, Any]:
        return self.service.health()

    def search_duckduckgo(self, query: str, limit: int = 4) -> List[Dict[str, str]]:
        response = self.service.search(
            InternetRetrievalQuery(text=query[:1_000], maximumResults=max(1, min(limit, 4))),
            trigger="direct-endpoint",
        )
        return self._legacy_results(response)

    def search_wikipedia(self, query: str) -> List[Dict[str, str]]:
        """The former unapproved second network endpoint is no longer contacted."""
        _ = query
        return []

    def get_component_info(
        self,
        component_name: str,
        force_refresh: bool = False,
        retrieval_reason: str = "direct-endpoint",
    ) -> Dict[str, Any]:
        local = self._local_component_info(component_name)
        if local is not None:
            return local
        if not self.internet_enabled:
            return {
                "component": component_name,
                "searchResults": [],
                "specs": ComponentSpecExtractor.extract_specs(component_name, []),
                "citations": [],
                "retrieval": {
                    **self.health(),
                    "mode": "offline-no-local-evidence",
                    "internetUsed": False,
                },
            }
        response = self.service.search(
            InternetRetrievalQuery(text=component_name[:1_000], maximumResults=4),
            trigger=retrieval_reason,
            bypass_cache=force_refresh,
        )
        results = self._legacy_results(response)
        return {
            "component": component_name,
            "searchResults": results,
            "specs": ComponentSpecExtractor.extract_specs(component_name, results),
            "citations": citations_from_response(response),
            "retrieval": {
                **response_metadata(response),
                "mode": {
                    "complete": "internet-cache" if response.cacheHit else "internet-live",
                    "no-results": "internet-no-evidence",
                    "degraded": "internet-degraded",
                    "disabled": "offline-no-local-evidence",
                    "not-requested": "offline-no-local-evidence",
                }[response.status],
                "internetEnabled": self.internet_enabled,
                "internetUsed": response.networkAccessed,
            },
        }

    def _local_component_info(self, component_name: str) -> Dict[str, Any] | None:
        try:
            from engine.knowledge_graph import ComponentKnowledgeGraph

            info = ComponentKnowledgeGraph.get_component_info(component_name)
        except Exception as error:
            logger.warning("Local component lookup failed (code=%s)", type(error).__name__)
            return None
        if not info:
            return None
        voltage = info.get("operatingVoltage") or {}
        minimum = voltage.get("min")
        maximum = voltage.get("max")
        operating_voltage = (
            f"{minimum}V - {maximum}V"
            if minimum is not None and maximum is not None
            else None
        )
        current_draw = info.get("currentDraw_mA")
        full_name = info.get("fullName", component_name)
        required = ", ".join(info.get("requiredExternalComponents", []))
        return {
            "component": component_name,
            "searchResults": [
                {
                    "title": full_name,
                    "snippet": f"{full_name}: {required}",
                    "source": "VoltForge Local Knowledge Graph",
                    "url": "local://datasheet",
                }
            ],
            "specs": {
                "query": component_name,
                "operatingVoltage": operating_voltage,
                "supportedInterfaces": info.get("interfaces", []),
                "i2cAddresses": info.get("i2cAddress", []),
                "summarySnippet": (
                    f"{full_name} ({info.get('category', '')}). Interfaces: "
                    f"{', '.join(info.get('interfaces', []))}."
                    + (f" Current: {current_draw}mA." if current_draw is not None else "")
                ),
            },
            "citations": [
                {
                    "title": full_name,
                    "url": "local://knowledge-base",
                    "source": "VoltForge Offline Database",
                }
            ],
            "retrieval": {
                "mode": "local",
                "internetEnabled": self.internet_enabled,
                "internetUsed": False,
                "generationDependency": False,
            },
        }

    @staticmethod
    def _legacy_results(response: Any) -> List[Dict[str, str]]:
        return [
            {
                "title": item.title,
                "url": item.sourceUrl,
                "snippet": item.snippet,
                "source": item.sourceDomain,
                "retrievedAt": item.retrievedAt,
                "contentSha256": item.contentSha256,
            }
            for item in response.evidence
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = WebSearchEngine()
    print(json.dumps(engine.get_component_info("MPU6050")["specs"], indent=2))
