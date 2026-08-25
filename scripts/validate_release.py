from __future__ import annotations

import json
import tomllib
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    python_files = sorted(root.rglob("*.py"))
    json_files = sorted(root.rglob("*.json"))
    for path in python_files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "pass",
                "python_files": len(python_files),
                "json_files": len(json_files),
                "pyproject": "valid",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
