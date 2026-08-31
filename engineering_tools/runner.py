"""Deterministic, evidence-bound engineering checks for one project revision."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from electronics_corpus.store import ElectronicsCorpus, get_electronics_corpus
from engineering_tools.schema import (
    ENGINEERING_POLICY_ID,
    EngineeringAuthorityReport,
    EngineeringContractError,
    EngineeringReportSummary,
    EngineeringToolRun,
    canonical_json,
    load_engineering_policy,
    sha256_json,
    validate_engineering_report,
)
from task_schema.adapters import runtime_request_to_task_record


_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "UNKNOWN": 3, "INFO": 4}
_HIGH_ENERGY = re.compile(
    r"\b(mains|line voltage|120\s*v|230\s*v|240\s*v|high[- ]voltage|"
    r"lithium pack|traction battery|safety[- ]critical)\b",
    re.IGNORECASE,
)
_QUANTITY_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*"
    r"([a-zA-ZµμΩΩ]+)\s*$"
)
_QUANTITY_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    "voltage": (("voltageV", "logicVoltageV", "supplyVoltageV"), "V"),
    "current": (("currentA", "targetCurrentA", "supplyCurrentA"), "A"),
    "resistance": (("resistanceOhm", "resistanceOhms"), "ohm"),
    "power": (("powerW", "ratedPowerW"), "W"),
}
_UNIT_TABLE: dict[str, tuple[str, Decimal]] = {
    "v": ("voltage", Decimal("1")),
    "mv": ("voltage", Decimal("0.001")),
    "kv": ("voltage", Decimal("1000")),
    "a": ("current", Decimal("1")),
    "ma": ("current", Decimal("0.001")),
    "ua": ("current", Decimal("0.000001")),
    "µa": ("current", Decimal("0.000001")),
    "μa": ("current", Decimal("0.000001")),
    "w": ("power", Decimal("1")),
    "mw": ("power", Decimal("0.001")),
    "kw": ("power", Decimal("1000")),
    "ohm": ("resistance", Decimal("1")),
    "ohms": ("resistance", Decimal("1")),
    "ω": ("resistance", Decimal("1")),
    "Ω": ("resistance", Decimal("1")),
    "kohm": ("resistance", Decimal("1000")),
    "kohms": ("resistance", Decimal("1000")),
    "kω": ("resistance", Decimal("1000")),
    "mohm": ("resistance", Decimal("1000000")),
}


def _request_mapping(request: Any) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        return request.model_dump(mode="python")
    return dict(request)


def _clean(value: object, maximum: int = 200) -> str:
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()[:maximum]


def _slug(value: object, maximum: int = 100) -> str:
    normalized = re.sub(r"[^a-z0-9._:-]+", "-", str(value or "").casefold()).strip("-.")
    if len(normalized) < 3:
        normalized = "item-" + hashlib.sha256(str(value).encode()).hexdigest()[:12]
    return normalized[:maximum]


def _sha_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _hashable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonFinite": "nan" if math.isnan(value) else "positive-infinity" if value > 0 else "negative-infinity"}
    if isinstance(value, Mapping):
        return {str(key): _hashable(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_hashable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _sha_safe_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(_hashable(value))).hexdigest()


def _claims(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(item["property"]): item.get("value")
        for item in record.get("claims", [])
        if isinstance(item, Mapping) and item.get("property")
    }


def _record_revision(record: Mapping[str, Any]) -> str:
    revision = record.get("effectiveRevision")
    return str(revision.get("revision") if isinstance(revision, Mapping) else "unknown")


def _bounded_facts(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(value)[:30]:
        item = value[key]
        if isinstance(item, str):
            result[str(key)[:80]] = _clean(item, 500)
        elif item is None or isinstance(item, (bool, int)):
            result[str(key)[:80]] = item
        elif isinstance(item, float):
            if math.isfinite(item):
                result[str(key)[:80]] = item
        elif isinstance(item, list):
            result[str(key)[:80]] = [
                _clean(child, 300) if isinstance(child, str) else child
                for child in item[:20]
                if child is None or isinstance(child, (str, bool, int, float))
            ]
        elif isinstance(item, Mapping):
            result[str(key)[:80]] = _bounded_facts(item)
    return result


class _ToolBuilder:
    def __init__(
        self,
        descriptor: Mapping[str, Any],
        source_revision: str,
        *,
        status: str = "complete",
    ):
        self.tool_id = str(descriptor["toolId"])
        self.version = str(descriptor["version"])
        self.category = str(descriptor["category"])
        self.source_revision = source_revision
        self.status = status
        self.evidence: dict[str, dict[str, Any]] = {}
        self.findings: list[dict[str, Any]] = []
        self.calculations: list[dict[str, Any]] = []

    @property
    def finding_capacity(self) -> bool:
        return True

    @property
    def evidence_capacity(self) -> bool:
        return True

    @property
    def calculation_capacity(self) -> bool:
        return True

    def add_evidence(
        self,
        *,
        kind: str,
        authority: str,
        source_id: str,
        source_revision: str,
        locator: str,
        content: Any,
        facts: Mapping[str, Any],
    ) -> str:
        content_sha = (
            _sha_bytes(content)
            if isinstance(content, str)
            else _sha_safe_json(content)
        )
        identity = {
            "kind": kind,
            "sourceId": source_id,
            "sourceRevision": source_revision,
            "locator": locator,
            "contentSha256": content_sha,
        }
        evidence_id = f"engineering-evidence:{sha256_json(identity)[:24]}"
        self.evidence[evidence_id] = {
            "evidenceId": evidence_id,
            "evidenceKind": kind,
            "authority": authority,
            "sourceId": _slug(source_id, 180),
            "sourceRevision": _clean(source_revision, 200) or "unknown",
            "locator": _clean(locator, 500) or "unspecified",
            "contentSha256": content_sha,
            "facts": _bounded_facts(facts),
            "rawContentStored": False,
        }
        return evidence_id

    def add_corpus_evidence(
        self, record: Mapping[str, Any], properties: Sequence[str] = ()
    ) -> str:
        facts = _claims(record)
        if properties:
            facts = {key: facts.get(key) for key in properties if key in facts}
        facts.update(
            {
                "recordType": record.get("recordType"),
                "supportStatus": record.get("supportStatus"),
            }
        )
        return self.add_evidence(
            kind="curated-knowledge",
            authority="deterministic",
            source_id=str(record["recordId"]),
            source_revision=_record_revision(record),
            locator=f"electronics_corpus/{record.get('recordType')}",
            content=record,
            facts=facts,
        )

    def add_project_evidence(
        self,
        source_id: str,
        locator: str,
        content: Any,
        facts: Mapping[str, Any],
        *,
        kind: str = "project-state",
        authority: str = "deterministic",
    ) -> str:
        return self.add_evidence(
            kind=kind,
            authority=authority,
            source_id=source_id,
            source_revision=self.source_revision,
            locator=locator,
            content=content,
            facts=facts,
        )

    def add_finding(
        self,
        *,
        rule_id: str,
        severity: str,
        decision: str,
        summary: str,
        affected_ids: Iterable[str],
        evidence_refs: Iterable[str],
        fix_summary: str,
        blocking: bool = False,
        action_kind: str | None = None,
        action_payload: Mapping[str, Any] | None = None,
    ) -> None:
        affected = sorted({_clean(item, 200) or "project:current" for item in affected_ids})
        references = sorted(set(evidence_refs))
        identity = {
            "tool": self.tool_id,
            "rule": rule_id,
            "severity": severity,
            "affected": affected,
            "evidence": references,
            "summary": summary,
        }
        finding_id = f"engineering-finding:{sha256_json(identity)[:24]}"
        payload = dict(action_payload or {})
        fix_identity = {
            "findingId": finding_id,
            "summary": fix_summary,
            "actionKind": action_kind,
            "payload": payload,
        }
        self.findings.append(
            {
                "findingId": finding_id,
                "ruleId": _slug(rule_id),
                "ruleVersion": self.version,
                "category": self.category,
                "severity": severity,
                "decision": decision,
                "summary": _clean(summary, 2_000),
                "affectedProjectIds": affected,
                "evidenceRefs": references,
                "fix": {
                    "fixId": f"engineering-fix:{sha256_json(fix_identity)[:24]}",
                    "summary": _clean(fix_summary, 1_000),
                    "actionKind": action_kind,
                    "payload": payload,
                    "applicationMode": "proposal-only",
                    "requiresUserConfirmation": True,
                    "machineApplicable": action_kind is not None,
                    "evidenceRefs": references,
                },
                "blocking": blocking,
                "modelOverridePolicy": "prohibited" if blocking else "not-applicable",
            }
        )

    def add_calculation(
        self,
        *,
        kind: str,
        affected_ids: Iterable[str],
        inputs: Mapping[str, str],
        result: Mapping[str, str],
        evidence_refs: Iterable[str],
    ) -> None:
        identity = {
            "tool": self.tool_id,
            "kind": kind,
            "affected": sorted(set(affected_ids)),
            "inputs": dict(inputs),
            "result": dict(result),
        }
        self.calculations.append(
            {
                "calculationId": f"engineering-calculation:{sha256_json(identity)[:24]}",
                "calculationKind": kind,
                "affectedProjectIds": sorted(set(affected_ids)),
                "inputs": dict(inputs),
                "result": dict(result),
                "evidenceRefs": sorted(set(evidence_refs)),
                "exactDecimal": True,
            }
        )

    def build(self) -> EngineeringToolRun:
        self.findings.sort(
            key=lambda item: (
                _SEVERITY_ORDER[item["severity"]],
                item["ruleId"],
                item["findingId"],
            )
        )
        self.calculations.sort(key=lambda item: item["calculationId"])
        selected_findings: list[dict[str, Any]] = []
        referenced: set[str] = set()
        for finding in self.findings:
            candidate_refs = referenced | set(finding["evidenceRefs"])
            if len(selected_findings) >= 50 or len(candidate_refs) > 100:
                continue
            selected_findings.append(finding)
            referenced = candidate_refs
        selected_calculations: list[dict[str, Any]] = []
        for calculation in self.calculations:
            candidate_refs = referenced | set(calculation["evidenceRefs"])
            if len(selected_calculations) >= 100 or len(candidate_refs) > 100:
                continue
            selected_calculations.append(calculation)
            referenced = candidate_refs
        if referenced:
            evidence = [self.evidence[key] for key in sorted(referenced)]
        else:
            evidence = [self.evidence[key] for key in sorted(self.evidence)[:100]]
        actions = [
            {
                "actionKind": finding["fix"]["actionKind"],
                "payload": finding["fix"]["payload"],
            }
            for finding in selected_findings
            if finding["fix"]["machineApplicable"]
        ]
        if self.status == "not-applicable":
            summary = f"{self.tool_id} had no applicable input."
        else:
            blocking = sum(bool(item["blocking"]) for item in selected_findings)
            summary = (
                f"{self.tool_id} completed with {len(selected_findings)} finding(s), "
                f"{blocking} blocking finding(s), and {len(selected_calculations)} calculation(s)."
            )
        return EngineeringToolRun.model_validate(
            {
                "toolId": self.tool_id,
                "toolVersion": self.version,
                "category": self.category,
                "status": self.status,
                "summary": summary,
                "findings": selected_findings,
                "evidence": evidence,
                "calculations": selected_calculations,
                "approvedActions": actions,
                "compiled": False,
                "rawContentStored": False,
            }
        )


def _component_rows(raw: Mapping[str, Any], maximum: int) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(list(raw.get("components") or [])[:maximum]):
        if not isinstance(item, Mapping):
            continue
        identifier = _clean(item.get("id") or item.get("componentId"), 200)
        rows.append(
            {
                "id": identifier or f"component:{_sha_safe_json(item)[:16]}",
                "type": _clean(item.get("type"), 100).upper(),
                "name": _clean(item.get("name"), 200),
                "pins": [
                    dict(pin)
                    for pin in list(item.get("pins") or [])[:100]
                    if isinstance(pin, Mapping)
                ],
                "properties": dict(item.get("properties") or {})
                if isinstance(item.get("properties"), Mapping)
                else {},
                "raw": dict(item),
                "index": index,
            }
        )
    return sorted(rows, key=lambda item: (item["id"], _sha_safe_json(item["raw"])))


def _wire_rows(raw: Mapping[str, Any], maximum: int) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(list(raw.get("wires") or [])[:maximum]):
        if not isinstance(item, Mapping):
            continue
        identifier = _clean(item.get("id"), 200)
        rows.append(
            {
                "id": identifier or f"wire:{_sha_safe_json(item)[:16]}",
                "fromComponent": _clean(
                    item.get("fromComponent") or item.get("fromNodeId"), 200
                ),
                "fromPin": _clean(item.get("fromPin") or item.get("fromPinId"), 100),
                "toComponent": _clean(
                    item.get("toComponent") or item.get("toNodeId"), 200
                ),
                "toPin": _clean(item.get("toPin") or item.get("toPinId"), 100),
                "raw": dict(item),
                "index": index,
            }
        )
    return sorted(rows, key=lambda item: (item["id"], _sha_safe_json(item["raw"])))


def _corpus_record(corpus: ElectronicsCorpus, record_id: str) -> dict[str, Any] | None:
    value = corpus.by_id.get(record_id)
    return dict(value) if isinstance(value, Mapping) else None


def _safety_record(corpus: ElectronicsCorpus, suffix: str) -> dict[str, Any] | None:
    return _corpus_record(corpus, f"vf-knowledge-v1-safety.{suffix}")


def _adjacency(wires: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for wire in wires:
        left = str(wire.get("fromComponent") or "")
        right = str(wire.get("toComponent") or "")
        if left and right:
            graph[left].add(right)
            graph[right].add(left)
    return graph


def _run_circuit_netlist(
    descriptor: Mapping[str, Any],
    raw: Mapping[str, Any],
    revision: str,
    components: Sequence[dict[str, Any]],
    wires: Sequence[dict[str, Any]],
    corpus: ElectronicsCorpus,
) -> EngineeringToolRun:
    nets_value = raw.get("netlist")
    nets = (
        list((nets_value.get("nets") or nets_value.get("nodes") or []))
        if isinstance(nets_value, Mapping)
        else []
    )
    message = str(raw.get("message") or "")
    applicable = bool(components or wires or nets or _HIGH_ENERGY.search(message))
    builder = _ToolBuilder(
        descriptor, revision, status="complete" if applicable else "not-applicable"
    )
    if not applicable:
        return builder.build()

    component_map = {item["id"]: item for item in components}
    component_types = {item["id"]: item["type"] for item in components}
    graph = _adjacency(wires)

    for wire in wires:
        wire_ref: str | None = None

        def evidence_for_wire() -> str:
            nonlocal wire_ref
            if wire_ref is None:
                wire_ref = builder.add_project_evidence(
                    f"project-wire:{wire['id']}",
                    "project/wires",
                    wire["raw"],
                    {
                        "wireId": wire["id"],
                        "fromComponent": wire["fromComponent"],
                        "fromPin": wire["fromPin"],
                        "toComponent": wire["toComponent"],
                        "toPin": wire["toPin"],
                    },
                )
            return wire_ref

        missing = sorted(
            endpoint
            for endpoint in (wire["fromComponent"], wire["toComponent"])
            if not endpoint or endpoint not in component_map
        )
        if missing:
            builder.add_finding(
                rule_id="circuit.wire-endpoint-missing",
                severity="CRITICAL",
                decision="violation",
                summary="A wire references a component endpoint that is absent from this project revision.",
                affected_ids=[wire["id"], *missing],
                evidence_refs=[evidence_for_wire()],
                fix_summary="Reconnect or remove the wire after selecting existing component endpoints.",
                blocking=True,
            )
        if (
            wire["fromComponent"] == wire["toComponent"]
            and wire["fromPin"] == wire["toPin"]
            and wire["fromComponent"]
        ):
            builder.add_finding(
                rule_id="circuit.self-loop-endpoint",
                severity="HIGH",
                decision="violation",
                summary="A wire connects one pin directly back to the same pin.",
                affected_ids=[wire["id"], wire["fromComponent"]],
                evidence_refs=[evidence_for_wire()],
                fix_summary="Remove the self-loop or reconnect it to the intended distinct endpoint.",
                blocking=True,
            )
        for component_key, pin_key in (
            ("fromComponent", "fromPin"),
            ("toComponent", "toPin"),
        ):
            component = component_map.get(wire[component_key])
            if not component or not component["pins"]:
                continue
            declared = {
                _clean(pin.get("id") or pin.get("name"), 100).casefold()
                for pin in component["pins"]
            }
            if wire[pin_key].casefold() not in declared:
                component_ref = builder.add_project_evidence(
                    f"project-component:{component['id']}",
                    "project/components/pins",
                    component["raw"],
                    {
                        "componentId": component["id"],
                        "declaredPinCount": len(component["pins"]),
                    },
                )
                builder.add_finding(
                    rule_id="circuit.pin-endpoint-unknown",
                    severity="CRITICAL",
                    decision="violation",
                    summary="A wire references a pin that is not declared by its project component.",
                    affected_ids=[wire["id"], component["id"]],
                    evidence_refs=[evidence_for_wire(), component_ref],
                    fix_summary="Choose one of the component revision's declared pins and review the resulting net.",
                    blocking=True,
                )

    duplicate_groups: dict[tuple[tuple[str, str], tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    for wire in wires:
        endpoints = sorted(
            [
                (wire["fromComponent"], wire["fromPin"]),
                (wire["toComponent"], wire["toPin"]),
            ]
        )
        duplicate_groups[(endpoints[0], endpoints[1])].append(wire)
    for group in duplicate_groups.values():
        if len(group) < 2:
            continue
        group = group[:50]
        refs = [
            builder.add_project_evidence(
                f"project-wire:{wire['id']}",
                "project/wires",
                wire["raw"],
                {"wireId": wire["id"], "duplicateEndpointPair": True},
            )
            for wire in group
        ]
        builder.add_finding(
            rule_id="circuit.duplicate-wire",
            severity="WARNING",
            decision="warning",
            summary="Multiple project wires declare the same endpoint pair.",
            affected_ids=[wire["id"] for wire in group],
            evidence_refs=refs,
            fix_summary="Review the duplicate paths and retain only intentional connections.",
        )

    pin_to_nets: dict[str, list[str]] = defaultdict(list)
    for index, net in enumerate(nets[:300]):
        if not isinstance(net, Mapping):
            continue
        net_id = _clean(net.get("id") or net.get("name"), 200) or f"net:{index}"
        for pin in list(net.get("pins") or [])[:500]:
            pin_to_nets[_clean(pin, 200)].append(net_id)
    for pin, net_ids in sorted(pin_to_nets.items()):
        distinct = sorted(set(net_ids))
        if pin and len(distinct) > 1:
            ref = builder.add_project_evidence(
                f"project-net-pin:{_sha_bytes(pin)[:16]}",
                "project/netlist",
                {"pin": pin, "nets": distinct},
                {"pin": pin, "netIds": distinct},
            )
            builder.add_finding(
                rule_id="netlist.pin-in-multiple-nets",
                severity="CRITICAL",
                decision="violation",
                summary="One project pin is assigned to multiple distinct net identifiers.",
                affected_ids=[pin, *distinct],
                evidence_refs=[ref],
                fix_summary="Merge the intended net or remove the conflicting pin assignment before simulation or generation.",
                blocking=True,
            )

    if _HIGH_ENERGY.search(message):
        safety = _safety_record(corpus, "high-energy-boundary")
        refs = []
        if safety:
            refs.append(builder.add_corpus_evidence(safety))
        refs.append(
            builder.add_project_evidence(
                "project-request:high-energy",
                "request/message-classification",
                message,
                {"highEnergyBoundaryMatched": True},
            )
        )
        builder.add_finding(
            rule_id="safety.high-energy-boundary",
            severity="CRITICAL",
            decision="unknown",
            summary="The request enters a mains, high-energy, battery-fault, or safety-critical boundary that this local project evidence does not certify.",
            affected_ids=["project:current"],
            evidence_refs=refs,
            fix_summary="Stop unverified construction guidance and obtain applicable standards, exact ratings, protection, isolation, and qualified review.",
            blocking=True,
        )

    safety_led = _safety_record(corpus, "led-current-limit")
    safety_flyback = _safety_record(corpus, "inductive-flyback")
    for component in components:
        component_type = component["type"]
        neighbor_types = {component_types.get(item, "") for item in graph.get(component["id"], set())}
        missing_led_limit = "LED" in component_type and "OLED" not in component_type and not any(
            "RESISTOR" in item for item in neighbor_types
        )
        missing_flyback = any(
            token in component_type for token in ("RELAY", "MOTOR", "SOLENOID")
        ) and not any(
            token in neighbor
            for neighbor in neighbor_types
            for token in ("DIODE", "FLYBACK", "SNUBBER", "TVS")
        )
        if not missing_led_limit and not missing_flyback:
            continue
        component_ref = builder.add_project_evidence(
            f"project-component:{component['id']}",
            "project/components",
            component["raw"],
            {"componentId": component["id"], "componentType": component_type},
        )
        if missing_led_limit:
            refs = [component_ref]
            if safety_led:
                refs.append(builder.add_corpus_evidence(safety_led))
            builder.add_finding(
                rule_id="safety.led-series-current-limit-unverified",
                severity="HIGH",
                decision="unknown",
                summary="No series resistor or evidenced current driver is connected to this LED in the supplied topology.",
                affected_ids=[component["id"]],
                evidence_refs=refs,
                fix_summary="Select an exact LED and driver, then calculate and review current limiting from evidenced voltage, forward-voltage range, target current, and power margin.",
                blocking=True,
            )
        if missing_flyback:
            refs = [component_ref]
            if safety_flyback:
                refs.append(builder.add_corpus_evidence(safety_flyback))
            builder.add_finding(
                rule_id="safety.inductive-suppression-unverified",
                severity="CRITICAL",
                decision="unknown",
                summary="The inductive load has no connected, evidenced suppression path in the supplied topology.",
                affected_ids=[component["id"]],
                evidence_refs=refs,
                fix_summary="Verify whether the exact driver/load includes protection; otherwise select and review a correctly oriented, rated suppression path.",
                blocking=True,
            )
    return builder.build()


def _normalize_pin(value: object) -> str:
    pin = re.sub(r"\s+", "", str(value or "").upper())
    if pin.startswith("GND"):
        return "GND"
    if pin in {"3.3V", "3V3", "3V"}:
        return "3V3"
    if pin in {"5.0V", "5V"}:
        return "5V"
    if pin.isdigit():
        return "D" + pin
    return pin


def _pin_in_expression(pin: str, expression: str) -> bool:
    pin = _normalize_pin(pin)
    for part in expression.upper().split(","):
        part = part.strip()
        if "-" not in part:
            if pin == _normalize_pin(part):
                return True
            continue
        left, right = part.split("-", 1)
        match_left = re.fullmatch(r"([A-Z_]+)(\d+)", _normalize_pin(left))
        match_right = re.fullmatch(r"([A-Z_]+)(\d+)", _normalize_pin(right))
        match_pin = re.fullmatch(r"([A-Z_]+)(\d+)", pin)
        if (
            match_left
            and match_right
            and match_pin
            and match_left.group(1) == match_right.group(1) == match_pin.group(1)
            and int(match_left.group(2)) <= int(match_pin.group(2)) <= int(match_right.group(2))
        ):
            return True
    return False


def _capabilities(pin: str, mapping: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for expression, values in mapping.items():
        if _pin_in_expression(pin, str(expression)) and isinstance(values, list):
            result.update(str(item) for item in values)
    return result


def _board_component_ids(
    corpus: ElectronicsCorpus,
    components: Sequence[Mapping[str, Any]],
    board_record_id: str,
    board_type: str,
) -> set[str]:
    ids = set()
    for component in components:
        component_type = str(component.get("type") or "")
        if component_type.casefold() == board_type.casefold():
            ids.add(str(component["id"]))
            continue
        lookup = corpus.lookup("board", component_type)
        if lookup.status == "found" and lookup.records[0]["recordId"] == board_record_id:
            ids.add(str(component["id"]))
    return ids


def _run_board_pin(
    descriptor: Mapping[str, Any],
    raw: Mapping[str, Any],
    revision: str,
    components: Sequence[dict[str, Any]],
    wires: Sequence[dict[str, Any]],
    corpus: ElectronicsCorpus,
) -> EngineeringToolRun:
    builder = _ToolBuilder(descriptor, revision)
    board_type = _clean(raw.get("boardType") or "ARDUINO_UNO", 80)
    board_lookup = corpus.lookup("board", board_type)
    pin_lookup = corpus.lookup("pin-map", board_type)
    request_ref = builder.add_project_evidence(
        f"project-board:{board_type}",
        "project/boardType",
        board_type,
        {"requestedBoardType": board_type},
    )
    if board_lookup.status != "found" or pin_lookup.status != "found":
        reason = (
            board_lookup.reasonCode
            if board_lookup.status != "found"
            else pin_lookup.reasonCode
        )
        builder.add_finding(
            rule_id="board-pin.exact-variant-required",
            severity="CRITICAL",
            decision="unknown",
            summary="Board pin capabilities cannot be verified without one exact supported board and pin-map revision.",
            affected_ids=[f"board:{board_type}"],
            evidence_refs=[request_ref],
            fix_summary=f"Select an exact supported board variant and revision; lookup status is {reason}.",
            blocking=True,
        )
        return builder.build()

    board = board_lookup.records[0]
    pin_map = pin_lookup.records[0]
    board_ref = builder.add_corpus_evidence(
        board, ("logic-voltage-v", "mcu", "architecture")
    )
    pin_ref = builder.add_corpus_evidence(
        pin_map, ("board-record-id", "power-pins", "default-buses", "pin-capabilities")
    )
    pin_claims = _claims(pin_map)
    capability_map = pin_claims.get("pin-capabilities") or {}
    if not isinstance(capability_map, Mapping):
        capability_map = {}
    board_ids = _board_component_ids(
        corpus, components, str(board["recordId"]), board_type
    )
    for wire in wires:
        endpoints = [
            (wire["fromComponent"], wire["fromPin"], wire["toPin"]),
            (wire["toComponent"], wire["toPin"], wire["fromPin"]),
        ]
        for component_id, pin, peer_pin in endpoints:
            if component_id not in board_ids:
                continue
            capabilities = _capabilities(pin, capability_map)
            if "flash-reserved" in capabilities:
                wire_ref = builder.add_project_evidence(
                    f"project-wire:{wire['id']}",
                    "project/wires",
                    wire["raw"],
                    {"wireId": wire["id"], "boardPin": pin},
                )
                builder.add_finding(
                    rule_id="board-pin.flash-reserved-connected",
                    severity="CRITICAL",
                    decision="violation",
                    summary="A project wire uses a board pin reserved for the exact module's flash interface.",
                    affected_ids=[wire["id"], component_id, pin],
                    evidence_refs=[board_ref, pin_ref, wire_ref],
                    fix_summary="Move the connection to an evidenced available pin and review the firmware mapping.",
                    blocking=True,
                )
            if "3v3-only" in capabilities and _normalize_pin(peer_pin) == "5V":
                wire_ref = builder.add_project_evidence(
                    f"project-wire:{wire['id']}",
                    "project/wires",
                    wire["raw"],
                    {"wireId": wire["id"], "boardPin": pin, "peerPin": peer_pin},
                )
                safety = _safety_record(corpus, "logic-voltage")
                refs = [board_ref, pin_ref, wire_ref]
                if safety:
                    refs.append(builder.add_corpus_evidence(safety))
                builder.add_finding(
                    rule_id="board-pin.3v3-only-connected-to-5v",
                    severity="CRITICAL",
                    decision="violation",
                    summary="A 3.3V-only board interface is connected to an endpoint identified as 5V.",
                    affected_ids=[wire["id"], component_id, pin],
                    evidence_refs=refs,
                    fix_summary="Disconnect the incompatible voltage and review an evidenced supply or level-translation design.",
                    blocking=True,
                )

    code = str(raw.get("code") or "")
    defines = {name: _normalize_pin(pin) for name, pin in re.findall(r"#define\s+(\w+)\s+(\d+)", code)}
    writes = set(re.findall(r"\bdigitalWrite\s*\(\s*([A-Za-z_]\w*|\d+)", code))
    for token in sorted(writes):
        pin = defines.get(token, _normalize_pin(token))
        capabilities = _capabilities(pin, capability_map)
        if not capabilities.intersection({"input-only", "flash-reserved"}):
            continue
        source_ref = builder.add_project_evidence(
            f"firmware-active:{_sha_bytes(code)[:16]}",
            "firmware/digitalWrite",
            code,
            {"sourceSha256": _sha_bytes(code), "pinToken": token, "resolvedPin": pin},
            kind="firmware-source",
        )
        builder.add_finding(
            rule_id="board-pin.output-capability-conflict",
            severity="CRITICAL",
            decision="violation",
            summary="Firmware writes to a pin whose exact board map marks it input-only or flash-reserved.",
            affected_ids=[f"firmware:{_sha_bytes(code)[:16]}", pin],
            evidence_refs=[board_ref, pin_ref, source_ref],
            fix_summary="Select an evidenced output-capable pin and update both firmware and project wiring together.",
            blocking=True,
        )

    logic_voltage = _claims(board).get("logic-voltage-v")
    if isinstance(logic_voltage, (int, float)):
        for component in components:
            properties = component["properties"]
            candidate = properties.get("logicVoltageV")
            if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
                continue
            if float(candidate) <= float(logic_voltage) + 1e-12:
                continue
            component_ref = builder.add_project_evidence(
                f"project-component:{component['id']}",
                "project/components/properties/logicVoltageV",
                component["raw"],
                {
                    "componentId": component["id"],
                    "declaredLogicVoltageV": candidate,
                },
            )
            safety = _safety_record(corpus, "logic-voltage")
            refs = [board_ref, component_ref]
            if safety:
                refs.append(builder.add_corpus_evidence(safety))
            builder.add_finding(
                rule_id="board-pin.logic-voltage-exceeds-board",
                severity="CRITICAL",
                decision="violation",
                summary="A component's declared logic voltage exceeds the exact board logic voltage.",
                affected_ids=[component["id"], f"board:{board_type}"],
                evidence_refs=refs,
                fix_summary="Use an exact compatible component or add an evidenced level-translation design after reviewing both endpoint ratings.",
                blocking=True,
            )
    return builder.build()


def _firmware_sources(raw: Mapping[str, Any], maximum_files: int, maximum_chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(list(raw.get("files") or [])[:maximum_files]):
        if not isinstance(item, Mapping):
            continue
        content = str(item.get("content") or "")
        rows.append(
            {
                "id": f"firmware:{_sha_bytes(content)[:16]}",
                "filename": _clean(item.get("filename"), 255) or f"file-{index}.txt",
                "language": _clean(item.get("language"), 32),
                "content": content[:maximum_chars],
                "fullSha256": _sha_bytes(content),
                "truncated": len(content) > maximum_chars,
            }
        )
    code = str(raw.get("code") or "")
    if code and not any(item["fullSha256"] == _sha_bytes(code) for item in rows):
        rows.append(
            {
                "id": f"firmware:{_sha_bytes(code)[:16]}",
                "filename": "active-source.ino",
                "language": "cpp",
                "content": code[:maximum_chars],
                "fullSha256": _sha_bytes(code),
                "truncated": len(code) > maximum_chars,
            }
        )
    return sorted(rows[:maximum_files], key=lambda item: (item["filename"].casefold(), item["fullSha256"]))


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _run_firmware(
    descriptor: Mapping[str, Any],
    raw: Mapping[str, Any],
    revision: str,
    corpus: ElectronicsCorpus,
    limits: Mapping[str, Any],
) -> EngineeringToolRun:
    sources = _firmware_sources(
        raw,
        int(limits["maximumFirmwareFiles"]),
        int(limits["maximumFirmwareCharactersPerFile"]),
    )
    builder = _ToolBuilder(
        descriptor, revision, status="complete" if sources else "not-applicable"
    )
    if not sources:
        return builder.build()

    board_lookup = corpus.lookup("board", _clean(raw.get("boardType") or "ARDUINO_UNO", 80))
    board_record_id = (
        str(board_lookup.records[0]["recordId"]) if board_lookup.status == "found" else None
    )
    firmware_records = [record for record in corpus.records if record["recordType"] == "firmware-api"]
    for source in sources:
        code = source["content"]
        source_ref = builder.add_project_evidence(
            source["id"],
            "firmware/source",
            code,
            {
                "sourceSha256": source["fullSha256"],
                "charactersReviewed": len(code),
                "truncated": source["truncated"],
                "language": source["language"],
            },
            kind="firmware-source",
        )
        definitions: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for match in re.finditer(r"^\s*#define\s+(\w+)\s+(\d+)\b", code, re.MULTILINE):
            definitions[match.group(2)].append((match.group(1), _line_number(code, match.start())))
        for pin, declarations in sorted(definitions.items()):
            if len(declarations) < 2:
                continue
            names = sorted({item[0] for item in declarations})
            if len(names) < 2:
                continue
            builder.add_finding(
                rule_id="firmware.pin-definition-collision",
                severity="CRITICAL",
                decision="violation",
                summary=f"Multiple firmware symbols assign digital pin {pin}: {', '.join(names[:8])}.",
                affected_ids=[source["id"], *[f"symbol:{name}" for name in names]],
                evidence_refs=[source_ref],
                fix_summary="Reassign the conflicting symbol only after checking the exact board pin map and project wiring.",
                blocking=True,
            )

        used = set(re.findall(r"\b(?:digitalWrite|digitalRead)\s*\(\s*([A-Za-z_]\w*|\d+)", code))
        configured = set(re.findall(r"\bpinMode\s*\(\s*([A-Za-z_]\w*|\d+)", code))
        for pin in sorted(used - configured):
            builder.add_finding(
                rule_id="firmware.pin-mode-missing",
                severity="WARNING",
                decision="warning",
                summary=f"Firmware uses pin token {pin} for digital I/O without a visible pinMode declaration.",
                affected_ids=[source["id"], f"pin-token:{pin}"],
                evidence_refs=[source_ref],
                fix_summary="Review the intended direction and pull configuration, then add the matching pinMode in initialization code.",
            )

        for match in re.finditer(r"\bdelay\s*\(\s*(\d+)\s*\)", code):
            duration = int(match.group(1))
            if duration < 500:
                continue
            builder.add_finding(
                rule_id="firmware.blocking-delay",
                severity="WARNING",
                decision="warning",
                summary=f"A delay call blocks the firmware loop for {duration} ms.",
                affected_ids=[source["id"], f"line:{_line_number(code, match.start())}"],
                evidence_refs=[source_ref],
                fix_summary="Review timing requirements and replace the blocking wait with explicit state and monotonic elapsed-time checks when concurrency is required.",
            )

        loop_match = re.search(r"\bvoid\s+loop\s*\([^)]*\)\s*\{([\s\S]*)\}", code)
        if loop_match and re.search(r"\bEEPROM\.(?:write|put)\s*\(", loop_match.group(1)):
            builder.add_finding(
                rule_id="firmware.eeprom-write-in-loop",
                severity="HIGH",
                decision="warning",
                summary="Firmware contains an EEPROM write operation inside the main loop body.",
                affected_ids=[source["id"]],
                evidence_refs=[source_ref],
                fix_summary="Gate writes on a verified value change and review endurance requirements for the exact nonvolatile memory implementation.",
                blocking=True,
            )

        includes = set(re.findall(r"#include\s*[<\"]([^>\"]+)[>\"]", code))
        for record in firmware_records:
            claims = _claims(record)
            supported_boards = claims.get("supported-board-record-ids") or []
            if board_record_id not in supported_boards:
                continue
            symbols = [str(item) for item in claims.get("api-symbols") or []]
            include = str(claims.get("include") or "")
            used_symbols = [symbol for symbol in symbols if symbol and symbol in code]
            if (
                not used_symbols
                or include in includes
                or (include == "Arduino.h" and source["filename"].casefold().endswith(".ino"))
            ):
                continue
            api_ref = builder.add_corpus_evidence(
                record,
                ("framework", "version-policy", "include", "api-symbols", "supported-board-record-ids"),
            )
            builder.add_finding(
                rule_id="firmware.required-include-missing",
                severity="HIGH",
                decision="violation",
                summary=f"Firmware uses curated API symbol {used_symbols[0]} but does not include {include}.",
                affected_ids=[source["id"], f"api:{_slug(used_symbols[0])}"],
                evidence_refs=[source_ref, api_ref],
                fix_summary=f"Add the exact {include} dependency only through the project's pinned board/library toolchain, then compile the complete source.",
                blocking=True,
            )
    return builder.build()


def _diagnostic_text(item: Mapping[str, Any]) -> str:
    for key in ("message", "text", "diagnostic", "stderr", "output"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _run_compiler_feedback(
    descriptor: Mapping[str, Any],
    raw: Mapping[str, Any],
    revision: str,
    corpus: ElectronicsCorpus,
    limits: Mapping[str, Any],
) -> EngineeringToolRun:
    diagnostics = [
        dict(item)
        for item in list(raw.get("diagnostics") or [])[: int(limits["maximumDiagnostics"])]
        if isinstance(item, Mapping)
    ]
    builder = _ToolBuilder(
        descriptor, revision, status="complete" if diagnostics else "not-applicable"
    )
    if not diagnostics:
        return builder.build()
    records = [record for record in corpus.records if record["recordType"] == "compiler-diagnostic"]
    for index, diagnostic in enumerate(diagnostics):
        text = _diagnostic_text(diagnostic)
        if not text:
            text = json.dumps(diagnostic, sort_keys=True, default=str)
        severity_text = str(diagnostic.get("severity") or diagnostic.get("level") or "error").casefold()
        warning_only = "warn" in severity_text and "error" not in severity_text
        diagnostic_ref = builder.add_project_evidence(
            f"compiler-diagnostic:{_sha_bytes(text)[:16]}",
            "compiler/diagnostics",
            text,
            {
                "diagnosticIndex": index,
                "diagnosticSha256": _sha_bytes(text),
                "declaredSeverity": _clean(severity_text, 40),
                "compiled": bool(diagnostic.get("compiled") or diagnostic.get("success")),
            },
            kind="compiler-diagnostic",
        )
        matched: tuple[Mapping[str, Any], dict[str, Any]] | None = None
        for record in records:
            claims = _claims(record)
            pattern = claims.get("diagnostic-pattern")
            if isinstance(pattern, str) and re.search(pattern, text, re.IGNORECASE):
                matched = (record, claims)
                break
        if matched:
            record, claims = matched
            corpus_ref = builder.add_corpus_evidence(
                record, ("toolchain", "diagnostic-pattern", "causes", "remediation")
            )
            remediation = [str(item) for item in claims.get("remediation") or []]
            builder.add_finding(
                rule_id=f"compiler.{record['recordId'].split('.')[-1]}",
                severity="WARNING" if warning_only else "HIGH",
                decision="warning" if warning_only else "violation",
                summary=f"Compiler feedback matches the curated {record['subject']['name']} diagnostic class.",
                affected_ids=[
                    f"diagnostic:{_sha_bytes(text)[:16]}",
                    _clean(diagnostic.get("file"), 200) or "firmware:active",
                ],
                evidence_refs=[diagnostic_ref, corpus_ref],
                fix_summary=" ".join(remediation[:3])
                or "Review the exact compiler diagnostic and pinned toolchain before changing source.",
                blocking=not warning_only,
            )
        else:
            builder.add_finding(
                rule_id="compiler.unclassified-diagnostic",
                severity="WARNING" if warning_only else "UNKNOWN",
                decision="warning" if warning_only else "unknown",
                summary="Compiler feedback is present but does not match a curated deterministic diagnostic class.",
                affected_ids=[f"diagnostic:{_sha_bytes(text)[:16]}"],
                evidence_refs=[diagnostic_ref],
                fix_summary="Preserve the exact diagnostic, compiler/core version, target, and source location for review; do not invent a repair.",
                blocking=not warning_only,
            )
    run = builder.build()
    compiled = bool(diagnostics) and all(
        bool(item.get("compiled") or item.get("success")) for item in diagnostics
    )
    if compiled:
        run = run.model_copy(update={"compiled": True})
    return run


def _nonfinite_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, Mapping):
        for key, child in list(value.items())[:200]:
            paths.extend(_nonfinite_paths(child, f"{prefix}.{_clean(key, 80)}"))
    elif isinstance(value, list):
        for index, child in enumerate(value[:200]):
            paths.extend(_nonfinite_paths(child, f"{prefix}[{index}]"))
    return paths[:100]


def _run_simulation(
    descriptor: Mapping[str, Any],
    raw: Mapping[str, Any],
    revision: str,
) -> EngineeringToolRun:
    state = raw.get("simulationState")
    builder = _ToolBuilder(
        descriptor, revision, status="complete" if isinstance(state, Mapping) and state else "not-applicable"
    )
    if not isinstance(state, Mapping) or not state:
        return builder.build()
    counts = {
        key: len(value)
        for key in ("nodeVoltages", "branchCurrents", "componentPower", "pinStates")
        if isinstance((value := state.get(key)), Mapping)
    }
    state_ref = builder.add_project_evidence(
        f"simulation-state:{_sha_safe_json(state)[:16]}",
        "simulation/client-snapshot",
        state,
        {
            "source": "client-reported",
            "solverConverged": state.get("solverConverged"),
            "isSimulating": state.get("isSimulating"),
            "valueCounts": counts,
        },
        kind="simulation-state",
        authority="client-reported",
    )
    nonfinite = _nonfinite_paths(state)
    if nonfinite:
        builder.add_finding(
            rule_id="simulation.nonfinite-result",
            severity="CRITICAL",
            decision="violation",
            summary="The reported simulation snapshot contains non-finite numeric results.",
            affected_ids=[f"simulation-path:{_sha_bytes(path)[:12]}" for path in nonfinite],
            evidence_refs=[state_ref],
            fix_summary="Reject this snapshot, correct the solver/input failure, and rerun before using any reported value.",
            blocking=True,
        )
    if state.get("solverConverged") is False:
        builder.add_finding(
            rule_id="simulation.not-converged",
            severity="HIGH",
            decision="violation",
            summary="The client-reported simulation snapshot did not converge and cannot support final engineering conclusions.",
            affected_ids=["simulation:current-snapshot"],
            evidence_refs=[state_ref],
            fix_summary="Review topology, models, initial conditions, timestep, and solver diagnostics, then rerun to convergence.",
            blocking=True,
        )
    elif "solverConverged" not in state:
        builder.add_finding(
            rule_id="simulation.convergence-unknown",
            severity="UNKNOWN",
            decision="unknown",
            summary="The client-reported simulation snapshot does not declare convergence status.",
            affected_ids=["simulation:current-snapshot"],
            evidence_refs=[state_ref],
            fix_summary="Provide the solver convergence result and solver revision before relying on reported values.",
            blocking=True,
        )
    if state.get("isSimulating") is True:
        builder.add_finding(
            rule_id="simulation.snapshot-still-running",
            severity="WARNING",
            decision="warning",
            summary="The simulation was still running when this client snapshot was reported.",
            affected_ids=["simulation:current-snapshot"],
            evidence_refs=[state_ref],
            fix_summary="Capture and review a completed stable snapshot before making final decisions.",
        )
    return builder.build()


def _decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    normalized = value.normalize()
    text = format(normalized, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _quantity(
    properties: Mapping[str, Any], quantity: str
) -> tuple[Decimal | None, str | None, str | None]:
    fields, canonical_unit = _QUANTITY_FIELDS[quantity]
    for field in fields:
        if field not in properties:
            continue
        value = properties[field]
        if isinstance(value, bool):
            return None, field, "boolean is not a numeric quantity"
        if isinstance(value, (int, float, Decimal)):
            if isinstance(value, float) and not math.isfinite(value):
                return None, field, "non-finite quantity"
            try:
                return Decimal(str(value)), field, None
            except InvalidOperation:
                return None, field, "invalid decimal quantity"
        if isinstance(value, str):
            try:
                return Decimal(value.strip()), field, None
            except InvalidOperation:
                pass
            match = _QUANTITY_PATTERN.fullmatch(value)
            if not match:
                return None, field, "quantity string requires an explicit supported unit"
            unit_key = match.group(2).replace("Ω", "ω").casefold()
            unit = _UNIT_TABLE.get(unit_key)
            if unit is None or unit[0] != quantity:
                return None, field, f"unit is incompatible with {canonical_unit}"
            try:
                return Decimal(match.group(1)) * unit[1], field, None
            except InvalidOperation:
                return None, field, "invalid decimal quantity"
    return None, None, None


def _outside_tolerance(actual: Decimal, expected: Decimal, relative: Decimal, absolute: Decimal) -> bool:
    tolerance = max(abs(expected) * relative, absolute)
    return abs(actual - expected) > tolerance


def _run_units(
    descriptor: Mapping[str, Any],
    revision: str,
    components: Sequence[dict[str, Any]],
    numeric_policy: Mapping[str, Any],
) -> EngineeringToolRun:
    applicable = any(
        any(field in component["properties"] for fields, _ in _QUANTITY_FIELDS.values() for field in fields)
        for component in components
    )
    builder = _ToolBuilder(
        descriptor, revision, status="complete" if applicable else "not-applicable"
    )
    if not applicable:
        return builder.build()
    relative_ohm = Decimal(str(numeric_policy["ohmsLawRelativeTolerance"]))
    relative_power = Decimal(str(numeric_policy["powerRelativeTolerance"]))
    absolute = Decimal(str(numeric_policy["minimumAbsoluteTolerance"]))
    precision = int(numeric_policy["decimalPrecision"])
    with localcontext() as context:
        context.prec = precision
        for component in components:
            properties = component["properties"]
            values: dict[str, Decimal] = {}
            fields: dict[str, str] = {}
            errors: list[tuple[str, str, str]] = []
            for quantity in _QUANTITY_FIELDS:
                value, field, error = _quantity(properties, quantity)
                if field and error:
                    errors.append((quantity, field, error))
                elif field and value is not None:
                    values[quantity] = value
                    fields[quantity] = field
            component_ref = builder.add_project_evidence(
                f"project-component:{component['id']}",
                "project/components/properties",
                component["raw"],
                {
                    "componentId": component["id"],
                    "componentType": component["type"],
                    "quantityFields": sorted(fields.values()),
                    "invalidQuantityFields": sorted(item[1] for item in errors),
                },
            )
            for quantity, field, error in errors:
                builder.add_finding(
                    rule_id="units.invalid-or-incompatible-unit",
                    severity="HIGH",
                    decision="unknown",
                    summary=f"Property {field} cannot be interpreted as a finite {quantity} quantity: {error}.",
                    affected_ids=[component["id"], f"property:{field}"],
                    evidence_refs=[component_ref],
                    fix_summary="Provide a finite decimal value with the field's canonical unit or an explicit supported SI unit.",
                    blocking=True,
                )
            if values.get("resistance") is not None and values["resistance"] <= 0:
                builder.add_finding(
                    rule_id="units.nonpositive-resistance",
                    severity="CRITICAL",
                    decision="violation",
                    summary="A declared resistance used for deterministic calculation is zero or negative.",
                    affected_ids=[component["id"], f"property:{fields['resistance']}"],
                    evidence_refs=[component_ref],
                    fix_summary="Correct the component value and unit using the exact part/design requirement before calculation or simulation.",
                    blocking=True,
                )
            if {"voltage", "current", "resistance"}.issubset(values) and values["resistance"] > 0:
                expected_voltage = values["current"] * values["resistance"]
                builder.add_calculation(
                    kind="ohms-law",
                    affected_ids=[component["id"]],
                    inputs={
                        "voltageV": _decimal_text(values["voltage"]),
                        "currentA": _decimal_text(values["current"]),
                        "resistanceOhm": _decimal_text(values["resistance"]),
                    },
                    result={"calculatedVoltageV": _decimal_text(expected_voltage)},
                    evidence_refs=[component_ref],
                )
                if _outside_tolerance(values["voltage"], expected_voltage, relative_ohm, absolute):
                    builder.add_finding(
                        rule_id="calculation.ohms-law-inconsistent",
                        severity="HIGH",
                        decision="violation",
                        summary="Declared voltage, current, and resistance are inconsistent with Ohm's law beyond the versioned tolerance.",
                        affected_ids=[component["id"]],
                        evidence_refs=[component_ref],
                        fix_summary="Review the source quantities, units, operating point, and component model; correct the inconsistent declaration before use.",
                        blocking=True,
                    )
            if {"voltage", "current", "power"}.issubset(values):
                expected_power = values["voltage"] * values["current"]
                builder.add_calculation(
                    kind="electrical-power",
                    affected_ids=[component["id"]],
                    inputs={
                        "voltageV": _decimal_text(values["voltage"]),
                        "currentA": _decimal_text(values["current"]),
                        "powerW": _decimal_text(values["power"]),
                    },
                    result={"calculatedPowerW": _decimal_text(expected_power)},
                    evidence_refs=[component_ref],
                )
                if _outside_tolerance(values["power"], expected_power, relative_power, absolute):
                    builder.add_finding(
                        rule_id="calculation.power-inconsistent",
                        severity="HIGH",
                        decision="violation",
                        summary="Declared voltage, current, and power are inconsistent beyond the versioned tolerance.",
                        affected_ids=[component["id"]],
                        evidence_refs=[component_ref],
                        fix_summary="Review quantity signs, RMS/DC meaning, operating point, and units before correcting the declaration.",
                        blocking=True,
                    )
    return builder.build()


def _run_component_support(
    descriptor: Mapping[str, Any],
    revision: str,
    components: Sequence[dict[str, Any]],
    corpus: ElectronicsCorpus,
    board_type: str,
) -> EngineeringToolRun:
    builder = _ToolBuilder(
        descriptor, revision, status="complete" if components else "not-applicable"
    )
    if not components:
        return builder.build()
    for component in components:
        component_type = component["type"]
        if component_type.casefold() == board_type.casefold() or corpus.lookup(
            "board", component_type
        ).status == "found":
            continue
        project_ref = builder.add_project_evidence(
            f"project-component:{component['id']}",
            "project/components",
            component["raw"],
            {"componentId": component["id"], "componentType": component_type},
        )
        lookup = corpus.lookup("component", component_type)
        if lookup.status == "found" and lookup.records[0].get("supportStatus") == "supported":
            continue
        refs = [project_ref]
        for record in lookup.records[:4]:
            refs.append(
                builder.add_corpus_evidence(
                    record,
                    ("support-status", "manufacturer", "part-number", "terminals", "electrical-ratings"),
                )
            )
        variant_required = lookup.reasonCode == "variant-required" or bool(lookup.records)
        builder.add_finding(
            rule_id=(
                "component.exact-variant-required"
                if variant_required
                else "component.unsupported-or-missing-evidence"
            ),
            severity="HIGH" if variant_required else "UNKNOWN",
            decision="unknown",
            summary=(
                "This component label resolves only to conflicting, generic, or reference-only variants without exact usable ratings."
                if variant_required
                else "This component has no approved exact record in the current curated electronics corpus."
            ),
            affected_ids=[component["id"]],
            evidence_refs=refs,
            fix_summary="Select and record the exact manufacturer, part number, module revision, terminal map, and electrical ratings before authoritative use.",
            blocking=variant_required,
        )
    return builder.build()


def _resolve_revision(request: Any) -> tuple[str, str]:
    try:
        from context_compiler import ContextCompilerError, resolve_project_revision
    except ImportError:
        record = runtime_request_to_task_record(request)
        context = record["input"]["projectContext"]
        return str(context["sourceProjectRevision"]), str(context["revisionSource"])

    try:
        return resolve_project_revision(request)
    except (ContextCompilerError, EngineeringContractError):
        record = runtime_request_to_task_record(request)
        context = record["input"]["projectContext"]
        return str(context["sourceProjectRevision"]), str(context["revisionSource"])


def run_authoritative_engineering_checks(request: Any) -> EngineeringAuthorityReport:
    """Run all seven versioned tools without network or model access."""

    policy = load_engineering_policy()
    raw = _request_mapping(request)
    revision, revision_source = _resolve_revision(request)
    limits = policy["limits"]
    corpus = get_electronics_corpus()
    components = _component_rows(raw, int(limits["maximumComponents"]))
    wires = _wire_rows(raw, int(limits["maximumWires"]))
    descriptors = {item["category"]: item for item in policy["tools"]}
    runs = [
        _run_circuit_netlist(
            descriptors["circuit-netlist"], raw, revision, components, wires, corpus
        ),
        _run_board_pin(
            descriptors["board-pin"], raw, revision, components, wires, corpus
        ),
        _run_firmware(
            descriptors["firmware-static"], raw, revision, corpus, limits
        ),
        _run_compiler_feedback(
            descriptors["compiler-feedback"], raw, revision, corpus, limits
        ),
        _run_simulation(descriptors["simulation"], raw, revision),
        _run_units(
            descriptors["units-calculation"], revision, components, policy["numeric"]
        ),
        _run_component_support(
            descriptors["component-support"],
            revision,
            components,
            corpus,
            _clean(raw.get("boardType") or "ARDUINO_UNO", 80),
        ),
    ]
    findings = [finding for run in runs for finding in run.findings]
    blocking = sorted(finding.findingId for finding in findings if finding.blocking)
    summary = EngineeringReportSummary(
        toolRuns=len(runs),
        applicableToolRuns=sum(run.status != "not-applicable" for run in runs),
        findings=len(findings),
        blockingFindings=len(blocking),
        criticalFindings=sum(finding.severity == "CRITICAL" for finding in findings),
        unknownFindings=sum(finding.decision == "unknown" for finding in findings),
        calculations=sum(len(run.calculations) for run in runs),
        status="blocked" if blocking else "review-required" if findings else "pass",
    )
    identity = {
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "sourceProjectRevision": revision,
        "boardType": _clean(raw.get("boardType") or "ARDUINO_UNO", 80),
        "toolRuns": [run.model_dump(mode="json") for run in runs],
    }
    report = EngineeringAuthorityReport.model_validate(
        {
            "schemaVersion": 1,
            "contractVersion": policy["contractVersion"],
            "policyId": policy["policyId"],
            "policySha256": policy["policySha256"],
            "reportId": f"vf-engineering-report-v1-{sha256_json(identity)[:24]}",
            "sourceProjectRevision": revision,
            "revisionSource": revision_source,
            "boardType": identity["boardType"],
            "toolRuns": runs,
            "blockingFindingIds": blocking,
            "summary": summary,
            "rawProjectContentStored": False,
        }
    )
    validate_engineering_report(report)
    return report


def tool_events_from_report(
    report: EngineeringAuthorityReport | Mapping[str, Any],
) -> list[dict[str, Any]]:
    value = (
        report
        if isinstance(report, EngineeringAuthorityReport)
        else EngineeringAuthorityReport.model_validate(report)
    )
    def compact_finding(item: Any) -> dict[str, Any]:
        return {
            "findingId": item.findingId,
            "ruleId": item.ruleId,
            "severity": item.severity,
            "decision": item.decision,
            "summary": item.summary,
            "affectedProjectIds": item.affectedProjectIds,
            "evidenceRefs": item.evidenceRefs,
            "fix": {
                "summary": item.fix.summary,
                "actionKind": item.fix.actionKind,
                "payload": item.fix.payload,
                "requiresUserConfirmation": True,
            },
            "blocking": item.blocking,
            "modelOverridePolicy": item.modelOverridePolicy,
        }

    blocking_findings = sorted(
        (
            finding
            for run in value.toolRuns
            for finding in run.findings
            if finding.blocking
        ),
        key=lambda item: (_SEVERITY_ORDER[item.severity], item.findingId),
    )
    indexed_findings = blocking_findings[:8]
    indexed_evidence_ids = {
        evidence_id for finding in indexed_findings for evidence_id in finding.evidenceRefs
    }
    indexed_evidence = {
        item.evidenceId: item
        for run in value.toolRuns
        for item in run.evidence
        if item.evidenceId in indexed_evidence_ids
    }
    events = [
        {
            "name": "engineering-authority-index",
            "version": "1.0.0",
            "status": "complete",
            "authority": "deterministic",
            "summary": (
                f"Authoritative engineering report has {len(blocking_findings)} blocking "
                "finding(s); model override is prohibited."
            ),
            "evidence": {
                "schemaVersion": 1,
                "policyId": value.policyId,
                "policySha256": value.policySha256,
                "reportId": value.reportId,
                "issues": [compact_finding(item) for item in indexed_findings],
                "blockingFindingIds": [item.findingId for item in indexed_findings],
                "totalBlockingFindingCount": len(blocking_findings),
                "omittedBlockingFindingCount": max(0, len(blocking_findings) - 8),
                "evidenceSources": [
                    {
                        "evidenceId": item.evidenceId,
                        "evidenceKind": item.evidenceKind,
                        "authority": item.authority,
                        "sourceId": item.sourceId,
                        "sourceRevision": item.sourceRevision,
                        "contentSha256": item.contentSha256,
                        "factsSha256": sha256_json(item.facts),
                        "rawContentStored": False,
                    }
                    for item in (indexed_evidence[key] for key in sorted(indexed_evidence))
                ],
                "approvedActions": [],
                "criticalDecisionAuthority": "deterministic-tool",
                "modelOverrideAllowed": False,
                "rawContentStored": False,
            },
        }
    ]
    for run in value.toolRuns:
        if run.status == "not-applicable":
            continue
        findings = [compact_finding(item) for item in run.findings]
        blocking = sorted(item.findingId for item in run.findings if item.blocking)
        evidence_sources = [
            {
                "evidenceId": item.evidenceId,
                "evidenceKind": item.evidenceKind,
                "authority": item.authority,
                "sourceId": item.sourceId,
                "sourceRevision": item.sourceRevision,
                "contentSha256": item.contentSha256,
                "factsSha256": sha256_json(item.facts),
                "rawContentStored": False,
            }
            for item in run.evidence
        ]
        simulation_facts = next(
            (
                item.facts
                for item in run.evidence
                if item.evidenceKind == "simulation-state"
            ),
            {},
        )
        events.append(
            {
                "name": run.toolId,
                "version": run.toolVersion,
                "status": "complete" if run.status == "complete" else "unavailable",
                "authority": "deterministic",
                "summary": run.summary,
                "evidence": {
                    "schemaVersion": 1,
                    "policyId": value.policyId,
                    "policySha256": value.policySha256,
                    "reportId": value.reportId,
                    "toolId": run.toolId,
                    "toolVersion": run.toolVersion,
                    "category": run.category,
                    "issues": findings,
                    "evidenceSources": evidence_sources,
                    "calculations": [
                        item.model_dump(mode="json") for item in run.calculations
                    ],
                    "approvedActions": [
                        item.model_dump(mode="json") for item in run.approvedActions
                    ],
                    "blockingFindingIds": blocking,
                    "criticalDecisionAuthority": "deterministic-tool",
                    "modelOverrideAllowed": False if blocking else None,
                    "compiled": run.compiled,
                    "rawContentStored": False,
                    **(
                        {
                            "source": "client-reported",
                            "solverConverged": simulation_facts.get("solverConverged"),
                            "isSimulating": simulation_facts.get("isSimulating"),
                        }
                        if run.category == "simulation"
                        else {}
                    ),
                },
            }
        )
    return events


def render_authoritative_summary(
    report: EngineeringAuthorityReport | Mapping[str, Any], maximum_findings: int = 8
) -> str:
    value = (
        report
        if isinstance(report, EngineeringAuthorityReport)
        else EngineeringAuthorityReport.model_validate(report)
    )
    findings = sorted(
        (finding for run in value.toolRuns for finding in run.findings),
        key=lambda item: (_SEVERITY_ORDER[item.severity], item.findingId),
    )
    lines = [
        "Authoritative deterministic engineering checks",
        (
            f"Status: {value.summary.status}; {value.summary.findings} finding(s), "
            f"{value.summary.blockingFindings} blocking."
        ),
    ]
    if not findings:
        lines.append(
            "The checks applicable to this request produced no finding; this is not a hardware certification."
        )
    for finding in findings[:maximum_findings]:
        lines.append(f"- [{finding.severity}] {finding.summary}")
        lines.append(f"  Reviewable fix: {finding.fix.summary}")
    omitted = max(0, len(findings) - maximum_findings)
    if omitted:
        lines.append(f"- {omitted} additional typed finding(s) are available in tool evidence.")
    if value.summary.blockingFindings:
        lines.append(
            "Blocking findings are authoritative and cannot be overridden by model text or applied automatically."
        )
    return "\n".join(lines)


def engineering_tools_health() -> dict[str, Any]:
    try:
        policy = load_engineering_policy()
        corpus = get_electronics_corpus()
        return {
            "ready": True,
            "code": "READY",
            "policyId": policy["policyId"],
            "policySha256": policy["policySha256"],
            "contractVersion": policy["contractVersion"],
            "toolCount": len(policy["tools"]),
            "tools": [
                {"toolId": item["toolId"], "version": item["version"]}
                for item in policy["tools"]
            ],
            "corpusVersion": corpus.catalog["version"],
            "criticalModelOverrideAllowed": False,
            "rawProjectContentStored": False,
        }
    except Exception as error:
        return {
            "ready": False,
            "code": getattr(error, "code", "ENGINEERING_TOOLS_UNAVAILABLE"),
            "policyId": ENGINEERING_POLICY_ID,
            "toolCount": 7,
            "criticalModelOverrideAllowed": False,
            "rawProjectContentStored": False,
        }
