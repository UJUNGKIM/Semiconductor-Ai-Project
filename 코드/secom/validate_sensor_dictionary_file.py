"""Validate and optionally install an owner-completed SECOM sensor dictionary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sensor_semantics import validate_sensor_dictionary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    output_dir = project / "결과물" / "secom" / "sensor_semantics"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input, keep_default_na=False)
    normalized, report = validate_sensor_dictionary(frame)
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.install:
        if report["validation_status"] != "VERIFIED":
            raise ValueError("590개 전체가 VERIFIED인 사전만 운영 경로에 설치할 수 있습니다.")
        installed = project / "데이터" / "SECOM 데이터셋" / "sensor_dictionary.csv"
        normalized.to_csv(installed, index=False, encoding="utf-8-sig")
        print(f"installed={installed}")
    print(
        f"status={report['validation_status']} verified={report['verified_count']}/590 "
        f"digest={report['dictionary_digest']}"
    )


if __name__ == "__main__":
    main()
