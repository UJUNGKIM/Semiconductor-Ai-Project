"""Build an aggregate-only synthetic monitoring replay."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from monitoring_log import REQUIRED,analyze_monitoring_history
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/"결과물"/"secom"/"batch_safety_gate"/"gate_stress_validation.csv"; OUT=ROOT/"결과물"/"secom"/"monitoring_demo"
def main():
    OUT.mkdir(parents=True,exist_ok=True); src=pd.read_csv(SRC).head(21); times=pd.date_range("2026-01-01",periods=len(src),freq="h",tz="Asia/Seoul"); rows=[]
    for i,x in src.iterrows():
        status=str(x.status); rate=float(x.model_disagreement_rate)
        one=round(rate*314)
        rows.append(dict(log_schema_version=3,recorded_at=times[i].isoformat(),batch_id=f"demo-{i+1:03d}",source_name="synthetic_replay",profile="balanced_f2",row_count=314,both_models_defect=0,one_model_defect=one,both_models_normal=314-one,defect_alert_rate=one/314,model_disagreement_rate=one/314,ood_any_rate=float(x.ood_any_rate),ood_severe_rate=float(x.ood_severe_rate),gate_status=status,automatic_decision_allowed=bool(x.automatic_decision_allowed),review_target_count=314 if status=="STOP" else 0,input_digest=f"{i:064d}",catboost_model_hash="demo",xgboost_model_hash="demo",outcome_confirmed=False,confirmed_defects=None,operator_id="",note=f"합성 재생: {x.scenario_korean}; 판정별 건수·검토 수·해시는 설명용이며 실측 아님",confirmed_alerted_defects=None,feedback_evidence="none"))
    history=pd.DataFrame(rows,columns=REQUIRED); history.to_csv(OUT/"demo_monitoring_log.csv",index=False,encoding="utf-8-sig"); summary=analyze_monitoring_history(history); (OUT/"monitoring_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    ax=history.set_index(pd.to_datetime(history.recorded_at))[["ood_any_rate","ood_severe_rate","model_disagreement_rate"]].plot(figsize=(10,4)); ax.set_ylabel("rate"); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(OUT/"monitoring_trend.png",dpi=160); plt.close()
    (OUT/"summary.md").write_text(f"# SECOM 운영 모니터링 데모\n\n실제 운영 이력이 아니라 강건성 스트레스 결과를 시간순으로 재생한 합성 데모입니다. 센서 원본값은 저장하지 않습니다.\n\n- 배치 수: {summary['batch_count']}\n- STOP: {summary['stop_count']}\n- 최근 상태: {summary['status']}\n",encoding="utf-8"); print("모니터링 데모 생성 완료")
if __name__=="__main__":main()
