from typing import Any, Dict, List
import json
import os
from dotenv import load_dotenv
load_dotenv()

# LangChain OpenAI client (chat model)
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process

from agents import AGENT_PROMPTS

MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Build an Agent for a given key using the prompt template

def build_agent(agent_key: str) -> Agent:
    template = AGENT_PROMPTS.get(agent_key)
    if not template:
        raise ValueError(f"Unknown agent key: {agent_key}")
    llm = ChatOpenAI(model=MODEL_DEFAULT, temperature=0.0, api_key=OPENAI_API_KEY)
    return Agent(
        role=agent_key,
        goal=f"Generate structured JSON for {agent_key}",
        backstory=f"{agent_key} assistant that outputs JSON.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

# Define a Task for the given agent

def build_task(agent: Agent, input_text: str, prompt_template: str) -> Task:
    instructions = prompt_template + "\n\nINPUT:\n" + input_text
    return Task(
        description=f"Run {agent.role} with JSON output",
        agent=agent,
        expected_output="JSON array or object",
        input=instructions,
        output_json=True,
    )

# Single agent run using CrewAI

def run_crewai_single(agent_key: str, input_text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {agent_key: [{"error": "OPENAI_API_KEY missing"}]}
    prompt_template = AGENT_PROMPTS.get(agent_key)
    if not prompt_template:
        return {agent_key: [{"error": "Unknown agent key"}]}
    agent = build_agent(agent_key)
    task = build_task(agent, input_text, prompt_template)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    res = crew.kickoff()
    try:
        parsed = json.loads(res.raw) if hasattr(res, "raw") else json.loads(str(res))
    except Exception:
        parsed = [res.raw if hasattr(res, "raw") else str(res)]
    return {agent_key: parsed}

# Pipeline run using CrewAI (sequential)

def run_crewai_pipeline(customer_requirements: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY missing"}
    outputs: Dict[str, Any] = {}
    o1 = run_crewai_single("SYS1", customer_requirements)
    outputs.update(o1)
    sys1_text = json.dumps(o1.get("SYS1", []))
    o2 = run_crewai_single("SYS2", sys1_text)
    outputs.update(o2)
    sys2_text = json.dumps(o2.get("SYS2", []))
    o3 = run_crewai_single("Review", sys2_text)
    outputs.update(o3)
    o4 = run_crewai_single("SYS5", sys2_text)
    outputs.update(o4)
    return outputs
