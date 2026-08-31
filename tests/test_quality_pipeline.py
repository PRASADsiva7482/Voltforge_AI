from __future__ import annotations

from pathlib import Path

from tools.scan_quality_boundaries import _iter_spec_files


def test_quality_scanner_requires_declared_production_paths(tmp_path: Path) -> None:
    files, violations = _iter_spec_files(tmp_path, ("missing.py",))

    assert files == []
    assert violations == [
        {
            "code": "missing-production-boundary",
            "path": "missing.py",
            "marker": "required-path",
        }
    ]


def test_quality_scanner_only_collects_declared_code_suffixes(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "allowed.py").write_text("pass\n", encoding="utf-8")
    (source / "ignored.bin").write_bytes(b"provider")

    files, violations = _iter_spec_files(tmp_path, ("src",))

    assert violations == []
    assert files == [source / "allowed.py"]
