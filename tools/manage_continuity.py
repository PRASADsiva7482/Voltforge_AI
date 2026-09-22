"""Create or verify content-free VFAI-FU-010 continuity evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from operations.continuity import (  # noqa: E402
    build_registry_source_inventory,
    build_restore_receipt,
    validate_inventory,
    verify_backup_receipt,
    verify_owner_objectives_receipt,
    verify_restored_tree,
    verify_restore_receipt,
    verify_rotation_receipt,
)


DEFAULT_INVENTORY_PATH = AI_ROOT / "evaluation/reports/continuity-source-inventory-v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--generated-on", required=True)
    inventory.add_argument("--output", type=Path, default=DEFAULT_INVENTORY_PATH)
    verify_inventory = commands.add_parser("verify-inventory")
    verify_inventory.add_argument("--input", type=Path, default=DEFAULT_INVENTORY_PATH)
    backup = commands.add_parser("verify-backup")
    backup.add_argument("--manifest", type=Path, default=DEFAULT_INVENTORY_PATH)
    backup.add_argument("--receipt", type=Path, required=True)
    objectives = commands.add_parser("verify-objectives")
    objectives.add_argument("--receipt", type=Path, required=True)
    restore = commands.add_parser("verify-restore")
    restore.add_argument("--manifest", type=Path, default=DEFAULT_INVENTORY_PATH)
    restore.add_argument("--restored-root", type=Path, required=True)
    restore.add_argument("--restore-target-id", required=True)
    restore.add_argument("--restore-scheduler-sha256", required=True)
    restore.add_argument("--verified-on", required=True)
    restore.add_argument("--output", type=Path, required=True)
    restore_receipt = commands.add_parser("verify-restore-receipt")
    restore_receipt.add_argument("--manifest", type=Path, default=DEFAULT_INVENTORY_PATH)
    restore_receipt.add_argument("--receipt", type=Path, required=True)
    rotation = commands.add_parser("verify-rotation")
    rotation.add_argument("--receipt", type=Path, required=True)
    rotation.add_argument(
        "--secret-kind", choices=("service-token", "signing-trust"), required=True
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "inventory":
        result = build_registry_source_inventory(generated_on=args.generated_on)
        _write_json(result, args.output.resolve())
    elif args.command == "verify-inventory":
        result = _read_json(args.input.resolve())
        validate_inventory(result)
    elif args.command == "verify-backup":
        manifest = _read_json(args.manifest.resolve())
        result = verify_backup_receipt(_read_json(args.receipt.resolve()), manifest)
    elif args.command == "verify-objectives":
        result = verify_owner_objectives_receipt(_read_json(args.receipt.resolve()))
    elif args.command == "verify-restore":
        manifest = _read_json(args.manifest.resolve())
        verification = verify_restored_tree(manifest, args.restored_root.resolve())
        result = build_restore_receipt(
            verification,
            manifest,
            restore_target_id=args.restore_target_id,
            restore_scheduler_sha256=args.restore_scheduler_sha256,
            verified_on=args.verified_on,
        )
        _write_json(result, args.output.resolve())
    elif args.command == "verify-restore-receipt":
        manifest = _read_json(args.manifest.resolve())
        result = verify_restore_receipt(_read_json(args.receipt.resolve()), manifest)
    else:
        result = verify_rotation_receipt(
            _read_json(args.receipt.resolve()), secret_kind=args.secret_kind
        )
    summary = {
        key: result[key]
        for key in (
            "manifestKind",
            "manifestSha256",
            "entryCount",
            "totalBytes",
            "backupId",
            "restoreTargetId",
            "receiptSha256",
            "secretKind",
        )
        if key in result
    }
    print(json.dumps({"ok": True, "command": args.command, "result": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
