from __future__ import annotations

from hardware_coverage import get_component_coverage


def test_component_coverage_is_one_to_one_and_never_promotes_generic_labels() -> None:
    report = get_component_coverage()

    assert report["entryCount"] == 60
    assert report["summary"] == {
        "verified": 0,
        "variantRequired": 51,
        "simulationOnly": 9,
        "unsupported": 0,
        "distinctCuratedExactCandidates": 2,
    }
    assert report["genericLabelsMaySelectCandidate"] is False
    assert len({entry["componentType"] for entry in report["entries"]}) == 60


def test_physical_components_require_exact_identity_and_simulator_tools_are_separate() -> None:
    entries = {
        entry["componentType"]: entry for entry in get_component_coverage()["entries"]
    }

    for component_type in (
        "OPAMP_LM358",
        "VOLTAGE_REGULATOR_7805",
        "MOTOR_STEPPER",
        "RELAY_SINGLE",
        "PIR_SENSOR",
        "OLED_DISPLAY",
        "IC_74HC595",
    ):
        assert entries[component_type]["status"] == "variant-required"
        assert entries[component_type]["physicalIdentitySelected"] is False
        assert entries[component_type]["aiElectricalClaimsAllowed"] is False
        assert entries[component_type]["selectionRequirements"] == [
            "manufacturer",
            "orderable-part-number",
            "package-or-module-revision",
            "terminal-or-connector-pinout",
            "electrical-ratings-and-polarity",
        ]
    assert entries["GROUND"]["status"] == "simulation-only"
    assert entries["OSCILLOSCOPE"]["status"] == "simulation-only"


def test_generic_oled_exposes_candidates_without_selecting_one() -> None:
    entries = {
        entry["componentType"]: entry for entry in get_component_coverage()["entries"]
    }
    oled = entries["OLED_DISPLAY"]

    assert oled["candidateSelectionRequired"] is True
    assert {item["supportStatus"] for item in oled["curatedExactCandidates"]} == {
        "supported",
        "reference-only",
    }
    assert len(oled["curatedExactCandidates"]) == 2
    assert "recordId" not in oled
