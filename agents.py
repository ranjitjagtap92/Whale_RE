import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# NOTE: This file defines agent prompt templates. Integration with CrewAI/LangChain is provided in orchestrator.py.
# These prompt templates instruct the LLM how to output JSON schemas for downstream parsing.

AGENT_PROMPTS = {
  "SYS1": '''Transform customer requirements into SYS.1 requirements.

RULES:
- Each SYS.1 requirement MUST follow SMART (Specific, Measurable, Achievable, Relevant, Traceable).
- Format: "The system shall ..."
- Include Customer Req. ID, rationale, and reference to ISO/IEC/IEEE 29148 or ISO 26262 where applicable.
- Ensure all fields are filled (use "TBD" if uncertain).
- Domains must be selected from: ["SW","HW","System","Mechanical"]
- Priority must be: ["High","Medium","Low"]

Output JSON array schema:
[
  {
    "Traceability": {"Parent ID": "CUST_REQ-001", "Current ID": "SYS.1-001", "Next ID": "SYS.2-001"},
    "Customer Req. ID": "CUST_REQ-001",
    "Customer Requirement": "...",
    "SYS.1 Req. ID": "SYS.1-001",
    "SYS.1 Requirement": "The system shall ...",
  "Domain": "SW",
    "Priority": "High",
    "Rationale": "Rationale with reference to ISO 26262-6, Clause X.X",
    "Requirement Status": "Draft"
  }
]''',

  "SYS2": '''Convert SYS.1 requirements into SYS.2 requirements with technical metadata.

RULES:
- You may generate MORE THAN ONE SYS.2 requirement for a single SYS.1 when it contains multiple behaviors, modes, interfaces, or acceptance conditions. Split into distinct, atomic SYS.2 requirements.
- Each SYS.2 requirement MUST preserve SMART attributes from SYS.1.
- Specify how the subsystem will achieve the SYS.1 behavior (control logic, data flow, interfaces).
- Add interfaces, timing, diagnostic behavior, and ASIL level.
- Fault handling must explicitly state safe states or fallback behavior.
- Verification Method must be chosen from: ["Analysis","Inspection","Simulation","HIL Test","Road Test","Other"]
- Requirement Type must be: ["Functional","Non-Functional","Information"]
- Verification Level must be: ["System Qualification Test (SYS.5)", "System Integration Test (SYS.4)"]
- Always maintain bidirectional traceability with SYS.1.
- Ensure unique "SYS.2 Req. ID" values per output row. Use stable, incrementing IDs per SYS.1 (e.g., SYS.2-001a, SYS.2-001b) when splitting.

Output JSON array schema:
[
  {
    "Traceability": {"Parent ID": "SYS.1-001", "Current ID": "SYS.2-001", "Next ID": "SYS.5-001"},
    "SYS.1 Req. ID": "SYS.1-001",
    "SYS.1 Requirement": "...",
    "SYS.2 Req. ID": "SYS.2-001",
    "SYS.2 Requirement": "The system shall ...",
    "Type": "Functional",
    "Verification Method": "HIL Test",
    "Verification Criteria": {"Metric": "Response time", "Threshold": "<100 ms"},
  "Verification Level": "System Integration Test (SYS.4)",
    "Domain": "SW",
    "Requirement Status": "Draft"
  }
]''',

    "Review": '''Review SYS.2 requirements for compliance with ISO/IEC/IEEE 29148, ISO 26262, and SMART criteria.

RULES:
- Provide feedback for each SYS.2 requirement.
- Include the SYS.2 Requirement text in the output for reference.
- Explicitly check each SMART attribute.
- Provide a rewritten requirement if fixes are needed.
- Compliance check must be one of: ["Yes","Partial","No"]
- Severity must be one of: ["High","Medium","Low"]

Output JSON array schema:
[
  {
    "SYS.2 Req. ID": "SYS.2-001",
    "SYS.2 Requirement": "The system shall...",
    "Review Feedback": "...",
    "SMART Check": {"Specific":"Yes/No","Measurable":"Yes/No","Achievable":"Yes/No","Relevant":"Yes/No","Traceable":"Yes/No"},
    "SMART Fix Suggestion": "...",
    "Proposed Rewrite": "The system shall ...",
    "Compliance Check": "Partial",
    "Severity": "High",
    "Suggested Improvement": "Clarify measurable response time"
  }
]''',

    "SYS5": '''Generate SYS.5 test cases from SYS.2 requirements.

RULES:
- For each input SYS.2 requirement, generate at least ONE test case. When the requirement has multiple acceptance conditions, modes, or edge cases, generate multiple test cases to cover them.
- Test Level must be chosen from: ["Unit","Integration","System","Vehicle"]
- Link test cases to Safety Goals if ASIL is assigned.

Output JSON array schema:
[
  {
    "Traceability": {"Parent ID": "SYS.2-001", "Current ID": "SYS.5-001"},
    "SYS.2 Req. ID": "SYS.2-001",
    "SYS.2 Requirement": "...",
    "Test Case ID": "TC-001",
    "Description": "Verify that system maintains set speed under steady conditions",
    "Preconditions": "Vehicle ignition ON, cruise control active",
    "Test Steps": ["Step 1: Engage cruise control", "Step 2: Accelerate to 80 km/h", "Step 3: Release accelerator"],
    "Expected Result": "System maintains 80 km/h +/-2 km/h",
    "Pass/Fail Criteria": "Pass if speed deviation <=2 km/h for 60s",
    "Test Level": "System",
    "Safety Goal Link": "SG-001"
  }
]'''
}
