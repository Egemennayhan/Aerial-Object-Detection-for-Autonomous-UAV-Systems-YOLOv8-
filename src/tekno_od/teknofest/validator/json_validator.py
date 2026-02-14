"""TEKNOFEST 2025 - Havacilikta Yapay Zeka Yarismasi (Task-1: Object Detection)
JSON Format Gate (Offline Validation)

This module validates an output JSON file against the Object Detection JSON schema.
Use it locally/offline as a strict "format gate" before sending results.

Default schema path:
  src/tekno_od/teknofest/schema/od_schema.json

CLI:
  python -m tekno_od.teknofest.validator.json_validator --input outputs/sample_valid.json
  python -m tekno_od.teknofest.validator.json_validator --input <file> --schema <schema_path>

Exit codes:
  0 -> VALID
  1 -> INVALID (does not conform to schema)
  2 -> ERROR (missing/unreadable files, invalid JSON, invalid schema, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import ValidationError, SchemaError, validate


def _default_schema_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1] / "schema" / "od_schema.json"


def _read_json_file(path: Path) -> object:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"cannot read file '{path}': {e}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"invalid JSON in '{path}': {e.msg} (line {e.lineno}, col {e.colno})") from e


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tekno_od.teknofest.validator.json_validator",
        description="Validate a TEKNOFEST Task-1 Object Detection output JSON against the schema.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to output JSON file to validate (e.g., outputs/sample_valid.json)",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Optional: override schema path (default: src/tekno_od/teknofest/schema/od_schema.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    input_path = Path(args.input)
    schema_path = Path(args.schema) if args.schema else _default_schema_path()

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(2)
    if not schema_path.exists():
        print(f"ERROR: schema file not found: {schema_path}")
        sys.exit(2)

    try:
        instance = _read_json_file(input_path)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    try:
        schema = _read_json_file(schema_path)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    try:
        validate(instance=instance, schema=schema)
    except ValidationError as e:
        print(f"INVALID: {e.message}")
        sys.exit(1)
    except SchemaError as e:
        print(f"ERROR: invalid schema: {e.message}")
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: unexpected validation error: {e}")
        sys.exit(2)

    print("VALID")
    sys.exit(0)


if __name__ == "__main__":
    main()
