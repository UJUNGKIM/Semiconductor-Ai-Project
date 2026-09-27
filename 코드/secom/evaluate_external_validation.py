"""Evaluate an independently labelled future-lot CSV without retuning the frozen policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from external_validation import evaluate_external_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--declarations", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output_dir = args.output_dir or project / "결과물" / "secom" / "external_validation"
    protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    declarations = json.loads(args.declarations.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input, low_memory=False)
    summary, per_lot = evaluate_external_frame(frame, declarations, project, protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validated_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    per_lot.to_csv(output_dir / "per_lot_metrics.csv", index=False, encoding="utf-8-sig")
    print(
        f"status={summary['validation_status']} rows={summary['overall']['rows']} "
        f"lots={summary['distinct_lots']} overlap={summary['known_secom_exact_row_overlap_count']}"
    )


if __name__ == "__main__":
    main()
