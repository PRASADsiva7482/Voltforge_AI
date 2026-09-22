"""
Voltforge AI - Enterprise Monitoring, Security & API Hardening
Covers Features 381-400: Rate Limiting, Input Sanitization, CORS Hardening,
Request Logging, Performance Metrics, Error Tracking, Health Checks,
API Key Validation, Payload Size Limiting, Security Headers.
"""

import time
import logging
import hashlib
import re
from typing import Any, Callable, Dict, Optional
from collections import defaultdict

logger = logging.getLogger("voltforge-ai.security")


class RateLimiter:
    """Token-bucket rate limiter for API endpoints."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, list] = defaultdict(list)

    def is_allowed(self, client_id: str) -> Dict[str, Any]:
        now = time.time()
        bucket = self._buckets[client_id]
        # Remove expired entries
        bucket[:] = [t for t in bucket if now - t < self.window_seconds]

        if len(bucket) >= self.max_requests:
            retry_after = self.window_seconds - (now - bucket[0])
            return {
                "allowed": False,
                "remaining": 0,
                "retryAfter_s": round(retry_after, 1),
                "message": f"Rate limit exceeded. Try again in {round(retry_after)}s.",
            }

        bucket.append(now)
        return {
            "allowed": True,
            "remaining": self.max_requests - len(bucket),
            "retryAfter_s": 0,
        }


class InputSanitizer:
    """Sanitizes and validates incoming API payloads."""

    MAX_MESSAGE_LENGTH = 5000
    MAX_CODE_LENGTH = 50000
    MAX_COMPONENTS = 200
    MAX_WIRES = 500

    # XSS/injection patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__',
        r'subprocess',
        r'os\.system',
        r'import\s+os',
    ]

    @classmethod
    def sanitize_message(cls, message: str) -> Dict[str, Any]:
        if not message or not isinstance(message, str):
            return {"valid": False, "sanitized": "", "reason": "Message is empty or invalid."}

        if len(message) > cls.MAX_MESSAGE_LENGTH:
            return {"valid": False, "sanitized": "", "reason": f"Message exceeds {cls.MAX_MESSAGE_LENGTH} characters."}

        # Check for injection attempts
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return {"valid": False, "sanitized": "", "reason": "Potentially malicious input detected."}

        # Basic sanitization
        sanitized = message.strip()
        return {"valid": True, "sanitized": sanitized, "reason": None}

    @classmethod
    def validate_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        issues = []

        # Validate message
        msg = payload.get("message", "")
        if msg and len(msg) > cls.MAX_MESSAGE_LENGTH:
            issues.append(f"Message exceeds {cls.MAX_MESSAGE_LENGTH} chars.")

        # Validate code
        code = payload.get("code", "")
        if code and len(code) > cls.MAX_CODE_LENGTH:
            issues.append(f"Code exceeds {cls.MAX_CODE_LENGTH} chars.")

        # Validate components array
        components = payload.get("components", [])
        if len(components) > cls.MAX_COMPONENTS:
            issues.append(f"Components array exceeds {cls.MAX_COMPONENTS} items.")

        # Validate wires array
        wires = payload.get("wires", [])
        if len(wires) > cls.MAX_WIRES:
            issues.append(f"Wires array exceeds {cls.MAX_WIRES} items.")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }


class PerformanceMetrics:
    """Tracks API endpoint latency and throughput metrics."""

    def __init__(self):
        self._metrics: Dict[str, list] = defaultdict(list)
        self._request_count = 0
        self._error_count = 0

    def record_request(self, endpoint: str, duration_ms: float, status_code: int) -> None:
        self._request_count += 1
        self._metrics[endpoint].append({
            "duration_ms": duration_ms,
            "status": status_code,
            "timestamp": time.time(),
        })
        if status_code >= 400:
            self._error_count += 1

        # Keep only last 1000 per endpoint
        if len(self._metrics[endpoint]) > 1000:
            self._metrics[endpoint] = self._metrics[endpoint][-500:]

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "totalRequests": self._request_count,
            "totalErrors": self._error_count,
            "errorRate": round(self._error_count / max(1, self._request_count) * 100, 2),
            "endpoints": {},
        }

        for endpoint, records in self._metrics.items():
            durations = [r["duration_ms"] for r in records]
            if durations:
                summary["endpoints"][endpoint] = {
                    "count": len(records),
                    "avgLatency_ms": round(sum(durations) / len(durations), 2),
                    "p50_ms": round(sorted(durations)[len(durations) // 2], 2),
                    "p95_ms": round(sorted(durations)[int(len(durations) * 0.95)], 2) if len(durations) > 1 else durations[0],
                    "maxLatency_ms": round(max(durations), 2),
                }

        return summary


class SecurityHeaders:
    """Generates security response headers for API responses."""

    HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }

    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        return cls.HEADERS.copy()


class APIKeyValidator:
    """Simple API key validation for development environments."""

    def __init__(self, valid_keys: list = None):
        self.valid_keys = set(valid_keys or [])
        self.usage: Dict[str, int] = defaultdict(int)

    def add_key(self, key: str) -> None:
        self.valid_keys.add(key)

    def validate(self, key: str) -> Dict[str, Any]:
        if not key:
            return {"valid": False, "reason": "API key is required."}
        if key not in self.valid_keys:
            return {"valid": False, "reason": "Invalid API key."}
        self.usage[key] += 1
        return {"valid": True, "usage_count": self.usage[key]}


class HealthChecker:
    """Comprehensive system health check."""

    @classmethod
    def check(cls) -> Dict[str, Any]:
        import sys
        import os

        checks = {
            "status": "healthy",
            "python_version": sys.version.split()[0],
            "pid": os.getpid(),
            "uptime_info": "available",
            "checks": {},
        }

        # Check engine modules
        engine_modules = [
            "engine.fuzzy_resolver", "engine.physics_solver", "engine.pin_router",
            "engine.spice_engine", "engine.firmware_analyzer", "engine.knowledge_graph",
            "engine.troubleshooter", "engine.nl_synthesizer", "engine.bus_solver",
            "engine.power_manager", "engine.wireless_solver", "engine.digital_logic",
            "engine.analog_engine", "engine.deep_firmware", "engine.pcb_engine",
            "engine.context_memory",
        ]

        for mod in engine_modules:
            try:
                __import__(mod)
                checks["checks"][mod] = "OK"
            except ImportError as e:
                checks["checks"][mod] = f"FAIL: {e}"
                checks["status"] = "degraded"

        return checks


class ElectronicsSafetyGuard:
    """Enforces electrical safety constraints and prompt injection guardrails."""

    # Direct mains hazard patterns (110V-240V directly to GPIO, microcontroller, breadboard, without isolation)
    MAINS_HAZARDS = [
        re.compile(r"\b(?:110|120|220|230|240)\s*v(?:ac)?\b.*?\b(?:pin|gpio|arduino|esp32|pico|stm32|mcu|microcontroller|breadboard)\b", re.IGNORECASE),
        re.compile(r"\b(?:pin|gpio|arduino|esp32|pico|stm32|mcu|microcontroller|breadboard)\b.*?\b(?:110|120|220|230|240)\s*v(?:ac)?\b", re.IGNORECASE),
        re.compile(r"\b(?:connect|wire|plug|hook up)\b.*?\b(?:mains|wall outlet|220v|110v|240v)\b.*?\b(?:directly|direct|without isolation|without relay)\b", re.IGNORECASE),
        re.compile(r"\b(?:directly|direct|without isolation)\b.*?\b(?:mains|220v|110v|240v)\b", re.IGNORECASE),
    ]

    # Dead shorts on supply rails
    SHORT_CIRCUIT_HAZARDS = [
        re.compile(r"\b(?:short|bridge|connect directly)\b.*?\b(?:vcc|5v|3\.3v)\b.*?\b(?:gnd|ground)\b", re.IGNORECASE),
        re.compile(r"\b(?:connect|wire)\b.*?\b(?:vcc|5v|3\.3v)\b.*?\b(?:directly|straight)\b.*?\b(?:to|into)\b.*?\b(?:gnd|ground)\b", re.IGNORECASE),
    ]

    # Prompt injection and instruction override patterns
    INJECTION_PATTERNS = [
        re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts|rules)\b", re.IGNORECASE),
        re.compile(r"\bdisregard\s+(?:all\s+)?(?:safety\s+)?(?:rules|guidelines|instructions)\b", re.IGNORECASE),
        re.compile(r"\b(?:dan\s+mode|jailbreak|developer\s+mode|unrestricted\s+mode)\b", re.IGNORECASE),
        re.compile(r"\byou\s+are\s+no\s+longer\s+voltforge\b", re.IGNORECASE),
        re.compile(r"\bbypass\s+(?:safety|security|policy)\b", re.IGNORECASE),
        re.compile(r"\bunfiltered\s+(?:ai|mode|response)\b", re.IGNORECASE),
    ]

    # Safe isolation indicators (if present, user is asking how to safely isolate mains)
    SAFE_ISOLATION_INDICATORS = [
        "relay", "optocoupler", "opto-isolator", "solid state relay", "ssr",
        "galvanic isolation", "isolated driver", "isolation module", "safely control",
        "how to safely", "isolated gate", "flyback diode"
    ]

    @classmethod
    def check_prompt_safety(cls, prompt: str) -> Dict[str, Any]:
        """Inspect prompt for electrical hazards and prompt injection attempts."""
        if not prompt or not isinstance(prompt, str):
            return {"safe": True, "hazard_type": None, "refusal_message": None}

        lower = prompt.lower().strip()

        # Check prompt injection first
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(lower):
                return {
                    "safe": False,
                    "hazard_type": "PROMPT_INJECTION",
                    "refusal_message": (
                        "### Security Policy Notice\n\n"
                        "Instruction overrides, safety bypass requests, and jailbreak attempts are restricted. "
                        "VoltForge AI operates exclusively as an authoritative electronics engineering assistant, "
                        "grounded in verified physics, schematic analysis, and safe circuit design."
                    ),
                }

        # Check short circuit hazards
        for pattern in cls.SHORT_CIRCUIT_HAZARDS:
            if pattern.search(lower):
                return {
                    "safe": False,
                    "hazard_type": "SHORT_CIRCUIT_HAZARD",
                    "refusal_message": (
                        "### Critical Electrical Hazard: Direct Rail Short Circuit\n\n"
                        "**Safety Refusal**: Connecting a power rail (5V/3.3V/VCC) directly to Ground (GND) creates a zero-resistance dead short. "
                        "This will immediately trigger overcurrent shutdown, damage your power regulator or USB host port, and generate severe thermal stress.\n\n"
                        "**Correct Practice**: Current must always flow through an appropriate load impedance or active component (e.g. resistor, IC, sensor) before returning to GND."
                    ),
                }

        # Check mains hazard
        is_safe_query = any(ind in lower for ind in cls.SAFE_ISOLATION_INDICATORS)
        for pattern in cls.MAINS_HAZARDS:
            if pattern.search(lower):
                if is_safe_query and not any(term in lower for term in ("directly", "without isolation", "without relay", "raw mains")):
                    continue

                return {
                    "safe": False,
                    "hazard_type": "HIGH_VOLTAGE_MAINS_HAZARD",
                    "refusal_message": (
                        "### Critical Electrical Hazard: Direct Mains Voltage Exposure\n\n"
                        "**Safety Refusal**: Connecting 110V/220V/240V AC mains directly to microcontroller GPIO pins, breadboards, or unisolated circuits "
                        "poses immediate danger of fatal electric shock, catastrophic component explosion, and electrical fire.\n\n"
                        "**Mandatory Safety Architecture**:\n"
                        "1. **Galvanic Isolation**: Low-voltage logic (3.3V/5V) must be electrically decoupled from mains using optocoupler-isolated relays or solid-state relays (SSR) certified to UL508 or IEC 62368-1.\n"
                        "2. **Creepage & Clearance**: Maintain $\\ge 6.3\\text{mm}$ clearance between high-voltage AC traces and low-voltage DC signals on any PCB.\n"
                        "3. **Enclosure**: Mains connections must reside inside a non-conductive, grounded, fuse-protected enclosure."
                    ),
                }

        return {"safe": True, "hazard_type": None, "refusal_message": None}



if __name__ == "__main__":
    import json

    # Test rate limiter
    limiter = RateLimiter(max_requests=3, window_seconds=10)
    for i in range(5):
        result = limiter.is_allowed("client_1")
        print(f"Request {i + 1}: {json.dumps(result)}")

    # Test input sanitizer
    print("\n=== Input Sanitization ===")
    print(json.dumps(InputSanitizer.sanitize_message("Connect LED to pin 13"), indent=2))
    print(json.dumps(InputSanitizer.sanitize_message("<script>alert('xss')</script>"), indent=2))

    # Test health check
    print("\n=== Health Check ===")
    print(json.dumps(HealthChecker.check(), indent=2))
