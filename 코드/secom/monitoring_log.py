"""Aggregate-only operational monitoring helpers for SECOM."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Mapping
import pandas as pd

RATES=("defect_alert_rate","model_disagreement_rate","ood_any_rate","ood_severe_rate")
STATUSES={"PASS","CAUTION","STOP","ROW_LEVEL_ONLY","NOT_EVALUATED"}
CORE_REQUIRED=("recorded_at","batch_id","source_name","profile","row_count","both_models_defect","one_model_defect","both_models_normal",*RATES,"gate_status","automatic_decision_allowed","review_target_count","input_digest","catboost_model_hash","xgboost_model_hash","outcome_confirmed","confirmed_defects","operator_id","note")
REQUIRED=("log_schema_version",*CORE_REQUIRED,"confirmed_alerted_defects","feedback_evidence")

def _digest(results):
    text=results.sort_index(axis=1).to_csv(index=False,float_format="%.12g")
    return hashlib.sha256(text.encode()).hexdigest()

def build_monitoring_record(results:pd.DataFrame,ood:pd.DataFrame,*,batch_id:str,source_name:str,profile:str,gate_result:Mapping|None,review_target_count:int,model_hashes:Mapping|None=None,recorded_at=None,outcome_confirmed=False,confirmed_defects=None,confirmed_alerted_defects=None,feedback_evidence=None,operator_id="",note=""):
    if not len(results) or len(results)!=len(ood): raise ValueError("예측과 OOD 결과의 길이를 확인하세요.")
    labels=results["종합 판정"]
    both=int((labels=="두 모델 모두 불량").sum()); one=int(labels.isin(("CatBoost만 불량","XGBoost만 불량")).sum()); normal=int((labels=="두 모델 모두 정상").sum())
    hashes=model_hashes or {}; status=str(gate_result["status"]) if gate_result else "NOT_EVALUATED"
    evidence=feedback_evidence or ("manual_aggregate" if outcome_confirmed else "none")
    return {"log_schema_version":3,"recorded_at":(pd.Timestamp.now(tz="Asia/Seoul") if recorded_at is None else pd.Timestamp(recorded_at)).isoformat(),"batch_id":str(batch_id).strip(),"source_name":Path(str(source_name)).name[:120],"profile":str(profile),"row_count":len(results),"both_models_defect":both,"one_model_defect":one,"both_models_normal":normal,"defect_alert_rate":(both+one)/len(results),"model_disagreement_rate":one/len(results),"ood_any_rate":float((ood["OOD 상태"]!="in_distribution").mean()),"ood_severe_rate":float((ood["OOD 상태"]=="out_of_distribution").mean()),"gate_status":status,"automatic_decision_allowed":bool(gate_result["automatic_decision_allowed"]) if gate_result else False,"review_target_count":int(review_target_count),"input_digest":_digest(results),"catboost_model_hash":str(hashes.get("CatBoost",hashes.get("catboost",""))),"xgboost_model_hash":str(hashes.get("XGBoost",hashes.get("xgboost",""))),"outcome_confirmed":bool(outcome_confirmed),"confirmed_defects":confirmed_defects if outcome_confirmed else None,"operator_id":str(operator_id)[:80],"note":str(note)[:500],"confirmed_alerted_defects":confirmed_alerted_defects if outcome_confirmed else None,"feedback_evidence":evidence}

def validate_monitoring_log(frame):
    missing=[c for c in CORE_REQUIRED if c not in frame];
    if missing: raise ValueError(f"모니터링 로그 필수 열 누락: {missing}")
    frame=frame.copy()
    if "log_schema_version" not in frame: frame["log_schema_version"]=1
    if "confirmed_alerted_defects" not in frame: frame["confirmed_alerted_defects"]=pd.NA
    if "feedback_evidence" not in frame: frame["feedback_evidence"]="legacy_unknown"
    out=frame.loc[:,REQUIRED].copy(); out["recorded_at"]=pd.to_datetime(out["recorded_at"],errors="raise",utc=True)
    if out["recorded_at"].isna().any(): raise ValueError("기록 시각이 누락됐습니다.")
    if out["batch_id"].isna().any() or out["batch_id"].astype(str).str.strip().eq("").any(): raise ValueError("batch_id는 비워둘 수 없습니다.")
    if (~out["gate_status"].isin(STATUSES)).any(): raise ValueError("알 수 없는 gate_status가 있습니다.")
    for c in RATES:
        out[c]=pd.to_numeric(out[c],errors="raise")
        if (~out[c].between(0,1)).any(): raise ValueError(f"{c}은 0~1 범위여야 합니다.")
    for c in ("row_count","both_models_defect","one_model_defect","both_models_normal","review_target_count"):
        values=pd.to_numeric(out[c],errors="raise")
        if values.isna().any() or (values<0).any() or (values%1!=0).any(): raise ValueError(f"{c}는 음수가 아닌 정수여야 합니다.")
        out[c]=values.astype(int)
    if (out["row_count"]<=0).any(): raise ValueError("row_count는 양수여야 합니다.")
    if (out[["both_models_defect","one_model_defect","both_models_normal"]].sum(axis=1)!=out["row_count"]).any(): raise ValueError("판정별 합계와 row_count가 다릅니다.")
    if (out["review_target_count"]>out["row_count"]).any(): raise ValueError("검토 수가 행 수를 초과합니다.")
    for c in ("automatic_decision_allowed","outcome_confirmed"):
        values=out[c].astype(str).str.lower()
        if (~values.isin(("true","false"))).any(): raise ValueError(f"{c}는 true/false여야 합니다.")
        out[c]=values.eq("true")
    if (out["automatic_decision_allowed"] & ~out["gate_status"].isin(("PASS","ROW_LEVEL_ONLY"))).any(): raise ValueError("미평가·경고·차단 배치는 자동판정 허용으로 기록할 수 없습니다.")
    counts=pd.to_numeric(out["confirmed_defects"],errors="raise")
    confirmed=out["outcome_confirmed"]
    if (confirmed & (counts.isna() | (counts<0) | (counts>out["row_count"]) | (counts%1!=0))).any(): raise ValueError("검수 불량 수를 확인하세요.")
    out["confirmed_defects"]=counts.where(confirmed)
    overlaps=pd.to_numeric(out["confirmed_alerted_defects"],errors="coerce")
    has_overlap=confirmed & overlaps.notna()
    alerts=out["both_models_defect"]+out["one_model_defect"]
    if (has_overlap & ((overlaps<0) | (overlaps%1!=0) | (overlaps>counts) | (overlaps>alerts))).any(): raise ValueError("경보에 포함된 실제 불량 수를 확인하세요.")
    out["confirmed_alerted_defects"]=overlaps.where(confirmed)
    versions=pd.to_numeric(out["log_schema_version"],errors="raise")
    if versions.isna().any() or (versions<1).any() or (versions%1!=0).any(): raise ValueError("log_schema_version이 올바르지 않습니다.")
    out["log_schema_version"]=versions.astype(int)
    allowed_evidence={"none","manual_aggregate","row_level_complete","oof_replay","legacy_unknown"}
    if (~out["feedback_evidence"].isin(allowed_evidence)).any(): raise ValueError("알 수 없는 feedback_evidence가 있습니다.")
    if ((~out["outcome_confirmed"]) & out["feedback_evidence"].ne("none") & out["feedback_evidence"].ne("legacy_unknown")).any(): raise ValueError("미확인 결과에는 검수 근거를 지정할 수 없습니다.")
    if (out["ood_severe_rate"]>out["ood_any_rate"]+1e-9).any(): raise ValueError("심각 OOD 비율은 전체 OOD 비율보다 클 수 없습니다.")
    for c,expected in (("model_disagreement_rate",out["one_model_defect"]/out["row_count"]),("defect_alert_rate",(out["both_models_defect"]+out["one_model_defect"])/out["row_count"])):
        if ((out[c]-expected).abs()>1e-8).any(): raise ValueError(f"{c}와 판정별 건수가 일치하지 않습니다.")
    return out

def append_monitoring_record(existing,record):
    added=pd.DataFrame([record])
    merged=added if existing is None or existing.empty else pd.concat([existing,added],ignore_index=True)
    return validate_monitoring_log(merged).drop_duplicates("batch_id",keep="last").sort_values("recorded_at").reset_index(drop=True)

def analyze_monitoring_history(frame):
    out=validate_monitoring_log(frame).sort_values("recorded_at")
    if out.empty:return {"status":"EMPTY","batch_count":0}
    last=out.tail(5); latest=str(out.iloc[-1]["gate_status"]); nonpass=int(last["gate_status"].isin(("CAUTION","STOP")).sum()); consecutive=0
    for value in reversed(out["gate_status"].tolist()):
        if value in ("PASS","ROW_LEVEL_ONLY"):break
        consecutive+=1
    return {"status":"CRITICAL" if latest=="STOP" else ("NOT_EVALUATED" if latest=="NOT_EVALUATED" else ("WARNING" if latest=="CAUTION" or nonpass>=2 else "STABLE")),"batch_count":len(out),"latest_gate_status":latest,"stop_count":int((out["gate_status"]=="STOP").sum()),"caution_count":int((out["gate_status"]=="CAUTION").sum()),"recent_nonpass_count":nonpass,"consecutive_nonpass":consecutive,"recent_ood_any_mean":float(last["ood_any_rate"].mean()),"recent_disagreement_mean":float(last["model_disagreement_rate"].mean())}

def monitoring_csv_bytes(frame):
    out=validate_monitoring_log(frame)
    # Prevent spreadsheet formula execution in untrusted filenames and notes.
    for c in ("batch_id","source_name","profile","operator_id","note"):
        out[c]=out[c].fillna("").astype(str).map(lambda s: "'"+s if s.lstrip().startswith(("=","+","-","@")) else s)
    return out.to_csv(index=False).encode("utf-8-sig")
