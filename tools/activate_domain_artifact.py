"""Retired approval shortcut. Importing or executing this module never writes artifacts."""

import json


def main() -> int:
    print(json.dumps({
        "code": "MODEL_APPROVAL_SHORTCUT_DISABLED",
        "message": "Publish a new immutable package with passing evaluation evidence, then use tools/manage_model_registry.py. Historical experimental artifacts cannot be relabelled by this command.",
        "changedFiles": 0,
    }))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
