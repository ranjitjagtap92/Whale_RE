import pandas as pd
from utils import export_csv, export_excel_styled, export_word_styled, export_pdf_styled, export_reqif, export_all_as_zip, build_traceability_matrix
from orchestrator import run_single, run_pipeline

# Sample dataframes
cust_sys1 = pd.DataFrame([
  {"Customer Req. ID":"CUST_REQ-001","Customer Requirement":"The vehicle shall start","SYS.1 Req. ID":"SYS.1-001","SYS.1 Requirement":"The system shall allow engine start","Requirement Status":"Draft"},
  {"Customer Req. ID":"CUST_REQ-002","Customer Requirement":"The vehicle shall stop","SYS.1 Req. ID":"SYS.1-002","SYS.1 Requirement":"The system shall enable braking","Requirement Status":"Approved"},
])

sys1_sys2 = pd.DataFrame([
  {"SYS.1 Req. ID":"SYS.1-001","SYS.1 Requirement":"The system shall allow engine start","SYS.2 Req. ID":"SYS.2-001","SYS.2 Requirement":"Start button input processed"},
  {"SYS.1 Req. ID":"SYS.1-002","SYS.1 Requirement":"The system shall enable braking","SYS.2 Req. ID":"SYS.2-002","SYS.2 Requirement":"Brake pedal input processed"},
])

review = pd.DataFrame([
  {"SYS.2 Req. ID":"SYS.2-001","Review Feedback":"Looks fine","Compliance Check":"Yes"},
  {"SYS.2 Req. ID":"SYS.2-002","Review Feedback":"Add clarity","Compliance Check":"Partial"},
])

sys2_sys5 = pd.DataFrame([
  {"SYS.2 Req. ID":"SYS.2-001","Test Case ID":"TC-001","Description":"Press start button","Test Steps":["Ignition on","Press start"],"Expected Result":"Engine starts"},
  {"SYS.2 Req. ID":"SYS.2-002","Test Case ID":"TC-002","Description":"Press brake","Test Steps":["Vehicle moving","Press brake"],"Expected Result":"Vehicle slows"},
])

# Build RTM
rtm = build_traceability_matrix(cust_sys1, sys1_sys2, review, sys2_sys5)
print("RTM rows, cols:", rtm.shape)

# Exporters produce bytes and non-empty sizes
assert len(export_csv(cust_sys1)) > 0
assert len(export_excel_styled(cust_sys1)) > 0
assert len(export_word_styled(cust_sys1)) > 0
assert len(export_pdf_styled(cust_sys1)) > 0
assert len(export_reqif(cust_sys1)) > 0
assert len(export_all_as_zip({"RTM": rtm})) > 0
print("Exporters OK")

# Orchestrator should not crash without API key
assert isinstance(run_single("SYS1", "test"), dict)
assert isinstance(run_pipeline("test"), dict)
print("Orchestrator OK (no key)")

print("All smoke tests passed.")
