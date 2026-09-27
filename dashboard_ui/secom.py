"""Pure helpers for the SECOM review workspace.

Keeping filtering, comparison, validation guidance, and audit decoding outside
``app.py`` makes them independently testable and keeps Streamlit rendering thin.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd


FILTER_KEYS = ("review", "all", "urgent", "disagreement", "ood")


def prepare_review_queue(queue: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready queue with a text-and-symbol priority label."""
    output = queue.copy()
    if "처리 권고" not in output:
        output["처리 권고"] = np.where(
            output["종합 판정"] == "두 모델 모두 정상",
            "자동 처리 후보",
            "모델 판정 검토",
        )
    action = output["처리 권고"].fillna("").astype(str)
    consensus = output["종합 판정"].fillna("").astype(str)
    output["우선도"] = np.select(
        (
            action.str.contains("보류", na=False) | consensus.eq("두 모델 모두 불량"),
            action.str.contains("OOD|경계", regex=True, na=False)
            | consensus.isin(("CatBoost만 불량", "XGBoost만 불량")),
        ),
        ("● 긴급", "◆ 확인"),
        default="○ 일반",
    )
    return output


def review_filter_options(queue: pd.DataFrame) -> dict[str, str]:
    """Return user-facing filter labels with live result counts."""
    action = queue["처리 권고"].fillna("").astype(str)
    consensus = queue["종합 판정"].fillna("").astype(str)
    counts = {
        "review": int((action != "자동 처리 후보").sum()),
        "all": len(queue),
        "urgent": int(consensus.eq("두 모델 모두 불량").sum()),
        "disagreement": int(
            consensus.isin(("CatBoost만 불량", "XGBoost만 불량")).sum()
        ),
        "ood": int(action.str.contains("OOD|경계", regex=True, na=False).sum()),
    }
    names = {
        "review": "검토 대상",
        "all": "전체",
        "urgent": "불량 긴급",
        "disagreement": "모델 불일치",
        "ood": "OOD·경계",
    }
    return {f"{names[key]} {counts[key]:,}": key for key in FILTER_KEYS}


def filter_review_queue(
    queue: pd.DataFrame, filter_key: str, search: str = ""
) -> pd.DataFrame:
    """Apply one semantic filter and a forgiving row/text search."""
    action = queue["처리 권고"].fillna("").astype(str)
    consensus = queue["종합 판정"].fillna("").astype(str)
    if filter_key == "review":
        mask = action != "자동 처리 후보"
    elif filter_key == "urgent":
        mask = consensus.eq("두 모델 모두 불량")
    elif filter_key == "disagreement":
        mask = consensus.isin(("CatBoost만 불량", "XGBoost만 불량"))
    elif filter_key == "ood":
        mask = action.str.contains("OOD|경계", regex=True, na=False)
    else:
        mask = pd.Series(True, index=queue.index)
    filtered = queue.loc[mask]
    term = str(search).strip().casefold()
    if term:
        searchable = [
            column
            for column in ("행 번호", "우선도", "처리 권고", "종합 판정", "입력 신뢰도")
            if column in filtered
        ]
        text = filtered[searchable].fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered.loc[text.str.casefold().str.contains(term, regex=False)]
    return filtered.reset_index(drop=True)


def build_profile_comparison(
    catboost_scores: np.ndarray,
    xgboost_scores: np.ndarray,
    bundles: Mapping[str, Mapping],
    profile_labels: Mapping[str, str],
) -> pd.DataFrame:
    """Compare all saved thresholds without rerunning either model."""
    rows = []
    for label, profile in profile_labels.items():
        cat = np.asarray(catboost_scores) >= float(
            bundles["CatBoost"]["operating_thresholds"][profile]
        )
        xgb = np.asarray(xgboost_scores) >= float(
            bundles["XGBoost"]["operating_thresholds"][profile]
        )
        rows.append(
            {
                "운영 기준": label,
                "두 모델 불량": int((cat & xgb).sum()),
                "모델 불일치": int((cat != xgb).sum()),
                "두 모델 정상": int((~cat & ~xgb).sum()),
                "판정 일치율": float((cat == xgb).mean()),
            }
        )
    return pd.DataFrame(rows)


def diagnosis_error_guidance(message: str) -> list[str]:
    """Translate parser/model failures into concrete recovery actions."""
    text = str(message)
    guidance = []
    if "590" in text or "누락 센서" in text or "알 수 없는 열" in text:
        guidance.extend(
            (
                "CSV는 feature_0부터 feature_589까지 590개 센서 열을 포함해야 합니다.",
                "헤더 없는 파일은 각 행에 숫자 590개를 공백으로 구분하세요.",
            )
        )
    if "숫자로 변환" in text:
        guidance.append("표시된 열의 문자·단위 기호를 제거하고 숫자 또는 빈 값만 남기세요.")
    if "UTF-8" in text or "CP949" in text:
        guidance.append("파일을 UTF-8 CSV 또는 CP949 텍스트로 다시 저장하세요.")
    if "데이터가 없습니다" in text or "데이터 행" in text:
        guidance.append("빈 행만 있는 파일이 아닌지 확인하고 센서 데이터 행을 한 개 이상 넣으세요.")
    if not guidance:
        guidance.append("입력 양식을 내려받아 열 이름과 데이터 형식을 비교한 뒤 다시 실행하세요.")
    guidance.append("ID·timestamp·label 열은 허용되며 진단 시 자동으로 제외됩니다.")
    return list(dict.fromkeys(guidance))


def audit_events_frame(
    ledger: pd.DataFrame, event_type: str | None = None
) -> pd.DataFrame:
    """Flatten aggregate audit payloads for human-readable dashboard history."""
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    selected = ledger.copy()
    if event_type:
        selected = selected.loc[selected["event_type"] == event_type]
    rows = []
    for _, event in selected.iterrows():
        payload = json.loads(str(event["payload_json"]))
        rows.append(
            {
                "sequence": int(event["sequence"]),
                "recorded_at": str(event["recorded_at"]),
                "event_type": str(event["event_type"]),
                "actor": str(event["actor"]),
                "batch_id": str(event["batch_id"]),
                **payload,
            }
        )
    return pd.DataFrame(rows)
