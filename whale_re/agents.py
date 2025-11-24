import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# NOTE: This file defines agent prompt templates. Integration with CrewAI/LangChain is provided in orchestrator.py.
# These prompt templates instruct the LLM how to output JSON schemas for downstream parsing.

AGENT_PROMPTS = {
		"SYS1": '''Transform customer requirements into SYS.1 requirements.

RULES:
- Each SYS.1 requirement MUST follow the SMART principle (Specific, Measurable, Achievable, Relevant, Traceable).
- Use format: The system shall ...
- Include Customer Req. ID and Rationale referencing standards where applicable.
- Output JSON array using this schema:
[
	{
		"Customer Req. ID": "CUST_REQ-001",
		"Customer Requirement": "...",    "SYS.1 Req. ID": "SYS.1-001",
		"SYS.1 Requirement": "The system shall ...",    "Domain": "",    "Priority": "",    "Rationale": "",    "Requirement Status": "Draft"
	}
]''',

		"SYS2": '''Convert SYS.1 requirements into SYS.2 requirements with technical metadata.
RULES:
- You may generate MORE THAN ONE SYS.2 requirement for a single SYS.1 when it contains multiple behaviors, modes, interfaces, or acceptance conditions. Split into distinct, atomic SYS.2 requirements.
- Preserve SMART attributes from SYS.1.
- Add interfaces, timing, fault handling, ASIL.
- Ensure unique "SYS.2 Req. ID" values per output row. Use stable incrementing IDs per SYS.1 (e.g., SYS.2-001a, SYS.2-001b) when splitting.
- Output JSON array schema:
[
	{
		"SYS.1 Req. ID": "SYS.1-001",
		"SYS.1 Requirement": "...",    "SYS.2 Req. ID": "SYS.2-001",
		"SYS.2 Requirement": "The system shall ...",    "Type": "Functional",    "Verification Method": "HIL Test, Road Test",    "Verification Criteria": "",    "Domain": "",    "ASIL": "",    "Requirement Status": "Draft"
	}
]''',

		"Review": '''Review SYS.2 requirements for compliance with ISO/IEC/IEEE 29148 and ISO 26262 and SMART criteria.
RULES:
- For each SYS.2, provide SMART check and fix suggestions.
- Output JSON array schema:
[
	{
		"SYS.2 Req. ID": "SYS.2-001",
		"Review Feedback": "...",
		"SMART Check": {"Specific":"Yes/No","Measurable":"Yes/No","Achievable":"Yes/No","Relevant":"Yes/No","Traceable":"Yes/No"},
		"SMART Fix Suggestion": "...",
		"Compliance Check": "Yes/Partial/No",
		"Suggested Improvement": "..."
	}
]''',

		"SYS5": '''Generate SYS.5 test cases from SYS.2 requirements.
Output JSON array schema:
[
	{
		"SYS.2 Req. ID": "SYS.2-001",
		"SYS.2 Requirement": "...",    "Test Case ID": "TC-001",
		"Description": "...",    "Preconditions": "...",    "Test Steps": ["step1","step2"],    "Expected Result": "...",    "Pass/Fail Criteria": "..."
	}
]'''
}

