"""Refresh the static dashboard from the latest Cricsheet IPL archive."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "clean data" / "ipl_dashboard_pro.html"
PIPELINE = ROOT / "ipl_pipeline_clean.py"
PROCESSED = ROOT / "data" / "processed"
CRICSHEET_EXTRACT = ROOT / "data" / "ipl_json"
CRICSHEET_ARCHIVE = ROOT / "data" / "ipl_json.zip"
CRICSHEET_MARKER = ROOT / "data" / ".cricsheet_source"


def read_csv(name: str) -> list[dict[str, object]]:
    with (PROCESSED / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            else:
                try:
                    row[key] = float(value) if "." in value else int(value)
                except (ValueError, TypeError):
                    pass
    return rows


def extract_data(html: str) -> dict[str, object]:
    match = re.search(r"const DATA = (.*?);\s*\n\s*// Color Palette", html, re.S)
    if not match:
        raise RuntimeError("Could not find the embedded dashboard dataset.")
    return json.loads(match.group(1))


def main() -> None:
    for path in (CRICSHEET_EXTRACT, CRICSHEET_ARCHIVE, CRICSHEET_MARKER):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for path in (ROOT / "data" / "matches.csv", ROOT / "data" / "deliveries.csv"):
        if path.exists():
            path.unlink()
    subprocess.run([sys.executable, str(PIPELINE)], cwd=ROOT, check=True)

    html = DASHBOARD.read_text(encoding="utf-8")
    data = extract_data(html)
    table_map = {
        "teams": "dashboard_team_performance",
        "batsmen": "dashboard_batsman_stats",
        "bowlers": "dashboard_bowler_stats",
        "venues": "dashboard_venue_analysis",
        "seasons": "dashboard_season_analysis",
        "toss": "dashboard_toss_analysis",
    }
    for dashboard_key, table_name in table_map.items():
        data[dashboard_key] = read_csv(table_name)
    data["overview"] = read_csv("dashboard_overview")[0]
    data["data_source"] = "Cricsheet.org IPL JSON"
    data["data_updated"] = date.today().isoformat()

    replacement = "const DATA = " + json.dumps(data, separators=(",", ":")) + ";"
    updated = re.sub(
        r"const DATA = .*?;(?=\s*\n\s*// Color Palette)",
        replacement,
        html,
        count=1,
        flags=re.S,
    )
    if updated == html:
        raise RuntimeError("Could not replace the embedded dashboard dataset.")
    DASHBOARD.write_text(updated, encoding="utf-8")
    print(f"Updated {DASHBOARD} from Cricsheet on {data['data_updated']}.")


if __name__ == "__main__":
    main()