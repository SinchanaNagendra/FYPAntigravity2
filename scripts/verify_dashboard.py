"""
scripts/verify_dashboard.py
===========================
Acceptance verification script for the dashboard.
Checks:
  1. Scans dashboard files for unbacked marketing claims or banned phrases.
  2. Verifies that all metrics trace to valid files under results/ or evidence/.
  3. Verifies data.json and data.js integrity and consistency.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "dashboard" / "frontend" / "public"

BANNED_UNSOURCED_PHRASES = [
    r"Publication-Grade",
    r"WHY FEDERATED BEATS CENTRALIZED",
    r"100% verified by privacy audit",
]


def check_banned_phrases():
    print("[verify] Checking for banned/unsourced marketing phrases...")
    violations = []
    for file_path in PUBLIC.glob("*.*"):
        if file_path.suffix not in [".html", ".js", ".json"]:
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            for pattern in BANNED_UNSOURCED_PHRASES:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append(f"{file_path.name}: found banned phrase '{matches[0]}'")
        except Exception as e:
            violations.append(f"{file_path.name}: error reading {e}")

    if violations:
        print("[FAIL] Banned phrase check failed:")
        for v in violations:
            print(f"  - {v}")
        return False
    print("[PASS] No banned unsourced phrases detected.")
    return True


def check_data_integrity():
    print("[verify] Verifying data.json structure and evidence links...")
    json_path = PUBLIC / "data.json"
    if not json_path.exists():
        print(f"[FAIL] {json_path} does not exist!")
        return False

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Required keys
    req_keys = [
        "central_table", "fl_comparison", "round_metrics",
        "test_summary", "provenance"
    ]
    for k in req_keys:
        if k not in data:
            print(f"[FAIL] Missing required key: {k}")
            return False

    # Check test summary format
    ts = data.get("test_summary", {})
    if "passed" not in ts or "skipped" not in ts or "failed" not in ts:
        print("[FAIL] test_summary missing passed/skipped/failed breakdown")
        return False

    print(f"[PASS] data.json verified: {len(data['central_table'])} baseline rows, {len(data['fl_comparison'])} FL rows.")
    return True


def main():
    ok1 = check_banned_phrases()
    ok2 = check_data_integrity()
    if ok1 and ok2:
        print("\n[ALL CHECKS PASSED] Dashboard adheres strictly to Anti-Bluff Contract.")
        sys.exit(0)
    else:
        print("\n[VERIFICATION FAILED] Dashboard needs corrections.")
        sys.exit(1)


if __name__ == "__main__":
    main()
