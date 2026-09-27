"""Build deterministic SECOM release manifest and readiness reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_readiness import (
    build_release_readiness,
    render_readiness_html,
    report_frame,
)

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "release_readiness"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="표·JSON·HTML·요약만 갱신하고 기존 PNG는 유지합니다.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report, manifest = build_release_readiness(PROJECT)
    (OUTPUT_DIR / "release_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame = report_frame(report)
    frame.to_csv(OUTPUT_DIR / "readiness_matrix.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "release_readiness.html").write_text(
        render_readiness_html(report, manifest), encoding="utf-8"
    )
    if not args.skip_plot:
        import matplotlib.pyplot as plt

        colors = {"PASS": "#2e7d32", "WARN": "#ed8b00", "BLOCK": "#c62828"}
        plot = frame.copy()
        plot["value"] = 1
        ax = plot.plot.barh(
            x="check_id",
            y="value",
            color=[colors[status] for status in plot["status"]],
            legend=False,
            figsize=(10, 6),
        )
        ax.set_xlim(0, 1); ax.set_xlabel(""); ax.set_xticks([]); ax.invert_yaxis()
        for index, status in enumerate(plot["status"]):
            ax.text(0.02, index, status, va="center", color="white", fontweight="bold")
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / "readiness_matrix.png", dpi=160); plt.close()
    blockers = [item for item in report["checks"] if item["status"] == "BLOCK"]
    actions = "\n".join(
        f"- {item['title']}: {item['required_action']}" for item in blockers
    )
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM 통합 릴리스 준비도\n\n"
        f"- Release ID: `{manifest['release_id']}`\n"
        f"- 대학생 대회 데모: **{report['demo_readiness']}**\n"
        f"- 실제 생산 배포: **{report['production_readiness']}**\n"
        f"- 생산 차단 항목: {report['production_blocker_count']}개\n\n"
        "## 생산 배포 전 필요한 조치\n\n"
        f"{actions}\n\n"
        "이 판정은 프로젝트 증거의 완전성을 확인하는 도구이며 성능 보증·규제 승인·"
        "실제 장비 안전 인증을 의미하지 않습니다.\n",
        encoding="utf-8",
    )
    print(
        f"릴리스 준비도 생성 완료: demo={report['demo_readiness']}, "
        f"production={report['production_readiness']}, release={manifest['release_id']}"
    )


if __name__ == "__main__":
    main()
