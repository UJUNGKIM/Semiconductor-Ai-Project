import sys,unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"코드"/"secom"))
from monitoring_log import analyze_monitoring_history,append_monitoring_record,build_monitoring_record,monitoring_csv_bytes,validate_monitoring_log
class Tests(unittest.TestCase):
 def record(self,b="b1",s="PASS"):
  r=pd.DataFrame({"종합 판정":["두 모델 모두 정상","CatBoost만 불량"]});o=pd.DataFrame({"OOD 상태":["in_distribution","review"]});return build_monitoring_record(r,o,batch_id=b,source_name="raw.csv",profile="balanced_f2",gate_result={"status":s,"automatic_decision_allowed":s=="PASS"},review_target_count=1,recorded_at="2026-01-01")
 def test_aggregate_record(self):
  r=self.record();self.assertEqual(r["model_disagreement_rate"],.5);self.assertEqual(len(r["input_digest"]),64)
 def test_replace_and_alert(self):
  f=append_monitoring_record(None,self.record());f=append_monitoring_record(f,self.record("b1","STOP"));self.assertEqual(len(f),1);self.assertEqual(analyze_monitoring_history(f)["status"],"CRITICAL")
 def test_invalid_rate_is_rejected(self):
  f=append_monitoring_record(None,self.record());f.loc[0,"ood_any_rate"]=1.2
  with self.assertRaises(ValueError):validate_monitoring_log(f)
 def test_csv_formula_is_escaped(self):
  r=self.record();r["note"]="=cmd";f=append_monitoring_record(None,r)
  self.assertIn("'=cmd",monitoring_csv_bytes(f).decode("utf-8-sig"))
 def test_dashboard_and_demo_artifacts(self):
  app=(ROOT/"app.py").read_text(encoding="utf-8");self.assertIn('"운영 모니터링"',app)
  for name in ("demo_monitoring_log.csv","monitoring_summary.json","monitoring_trend.png","summary.md"):
   self.assertTrue((ROOT/"결과물"/"secom"/"monitoring_demo"/name).is_file(),name)
if __name__=="__main__":unittest.main()
