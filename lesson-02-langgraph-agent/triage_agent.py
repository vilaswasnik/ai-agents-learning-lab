"""
LESSON 02 — LangGraph: a server-incident triage agent as a STATE MACHINE.

LangChain (lesson 01) is a straight line: prompt -> model -> answer.
Real agent work is NOT a straight line. It has steps, decisions, loops —
and places where a HUMAN must approve before anything risky happens.

LangGraph models that as a graph:

    START -> classify -> check -> plan -> approval --YES--> execute -> verify -> report -> END
                                              |
                                              +-------NO-----> report -> END

Key ideas to notice while reading:
- STATE    : one shared dictionary every node reads and updates.
- NODE     : a plain Python function: state in, partial state out.
- EDGE     : "after this node, go there". CONDITIONAL edges choose the
             next node by looking at the state (here: approved or not).
- The approval node is the guardrail — the same pattern as the voice-agent
  project: the agent may plan anything, but it cannot act without the human.

The tools return MOCK data (a pretend server), so this runs anywhere.

Run it:
    python lesson-02-langgraph-agent/triage_agent.py
    python lesson-02-langgraph-agent/triage_agent.py "db01 high CPU, load average 40"
    python lesson-02-langgraph-agent/triage_agent.py "app01 httpd service is down"
At the APPROVAL step, type y or n.
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from shared.llm import get_chat_model, provider_name

model = get_chat_model()


# --------------------------------------------------------------------------
# STATE — everything the agent knows, carried from node to node.
# --------------------------------------------------------------------------
class TriageState(TypedDict, total=False):
    incident: str        # what the operator typed
    incident_type: str   # disk / cpu / service / memory / general
    severity: str        # HIGH / MEDIUM / LOW
    facts: str           # what the tools found
    plan: str            # the LLM's proposed fix
    approved: bool       # the human's decision
    result: str          # what execute did
    verified: str        # what verify found afterwards
    report: str          # final plain-English summary


# --------------------------------------------------------------------------
# MOCK TOOLS — in real life these would SSH to a server or call an API.
# Here they return canned facts per incident type, so you can learn the
# agent pattern without touching a real machine.
# --------------------------------------------------------------------------
MOCK_FACTS = {
    "disk": "disk / = 97% full; /var/log = 40 GB (httpd logs not rotated for 21 days); httpd running; load normal",
    "cpu": "load average 38.5 on 8 cores; top process: java (PID 2214) at 640% CPU for 2 hours; memory OK; disk OK",
    "service": "httpd service: FAILED (exit code 1); port 80 free; config test: syntax OK; last log line: 'could not open error log file'",
    "memory": "RAM 98% used; top process: python worker at 41 GB and growing steadily for 6 hours; swap 100% used",
    "general": "uptime 212 days; disk 71%; load 2.1; memory 63%; no failed services",
}


def ask(task_prompt: str) -> str:
    """One helper so every node talks to the model the same way."""
    messages = [
        SystemMessage(content="You are an incident triage assistant for a Linux operations team."),
        HumanMessage(content=task_prompt),
    ]
    return str(model.invoke(messages).content)


# --------------------------------------------------------------------------
# NODES — each one does ONE job and returns the state keys it changed.
# --------------------------------------------------------------------------
def classify_node(state: TriageState) -> dict:
    raw = ask(f"TASK: classify\nIncident report: {state['incident']}\n"
              "Reply with exactly two lines: TYPE=<type> and SEVERITY=<level>.")
    parsed = dict(
        line.split("=", 1) for line in raw.strip().splitlines() if "=" in line
    )
    incident_type = parsed.get("TYPE", "general").strip().lower()
    severity = parsed.get("SEVERITY", "MEDIUM").strip().upper()
    print(f"\n[CLASSIFY] type={incident_type}  severity={severity}")
    return {"incident_type": incident_type, "severity": severity}


def check_node(state: TriageState) -> dict:
    facts = MOCK_FACTS.get(state["incident_type"], MOCK_FACTS["general"])
    print(f"[CHECK] {facts}")
    return {"facts": facts}


def plan_node(state: TriageState) -> dict:
    plan = ask(
        f"TASK: plan\nIncident type: {state['incident_type']}\nFacts: {state['facts']}\n"
        "Write a numbered fix plan, one step per line."
    )
    print(f"[PLAN]\n{plan}")
    return {"plan": plan}


def approval_node(state: TriageState) -> dict:
    # THE GUARDRAIL. Nothing executes until a human says yes.
    # (In production LangGraph you would use interrupt() + a checkpointer
    # so the pause survives a restart — same idea, stronger plumbing.)
    answer = input("\n>>> APPROVE this plan? [y/n]: ").strip().lower()
    approved = answer in ("y", "yes")
    print(f"[APPROVAL] {'approved' if approved else 'NOT approved — stopping here'}")
    return {"approved": approved}


def execute_node(state: TriageState) -> dict:
    steps = [s for s in state["plan"].splitlines() if s.strip()]
    done = []
    for step in steps:
        print(f"[EXECUTE] would run: {step}")
        done.append(f"done: {step}")
    return {"result": "; ".join(done)}


def verify_node(state: TriageState) -> dict:
    after = {
        "disk": "disk / = 61% full; httpd logs rotated; service healthy",
        "cpu": "load average 1.8 and stable for 5 minutes; java process normal",
        "service": "httpd active (running); responding on port 80 (HTTP 200)",
        "memory": "RAM 54% used; worker restarted cleanly; swap draining",
        "general": "all checks green",
    }
    verified = after.get(state["incident_type"], after["general"])
    print(f"[VERIFY] {verified}")
    return {"verified": verified}


def report_node(state: TriageState) -> dict:
    if state.get("approved"):
        summary = ask(
            f"TASK: report\nIncident: {state['incident']}\nType: {state['incident_type']}\n"
            f"Facts: {state['facts']}\nVerified after fix: {state.get('verified', '')}\n"
            "Write a 2-3 line summary for the operator."
        )
    else:
        summary = (
            "Plan was NOT approved by the operator, so no action was taken. "
            f"Incident: {state['incident']} ({state['incident_type']}, "
            f"severity {state['severity']}). Facts collected: {state['facts']}"
        )
    print(f"\n[REPORT]\n{summary}")
    return {"report": summary}


def route_after_approval(state: TriageState) -> str:
    """Conditional edge: the state decides where the graph goes next."""
    return "execute" if state.get("approved") else "report"


# --------------------------------------------------------------------------
# BUILD THE GRAPH — nodes + edges. Compare this with the ASCII diagram.
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify_node)
    graph.add_node("check", check_node)
    graph.add_node("plan", plan_node)
    graph.add_node("approval", approval_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "check")
    graph.add_edge("check", "plan")
    graph.add_edge("plan", "approval")
    graph.add_conditional_edges("approval", route_after_approval,
                                {"execute": "execute", "report": "report"})
    graph.add_edge("execute", "verify")
    graph.add_edge("verify", "report")
    graph.add_edge("report", END)
    return graph.compile()


def main() -> None:
    incident = " ".join(sys.argv[1:]) or "web01 disk is full, the website is very slow"
    print(f"[provider: {provider_name()}]")
    print(f"[INCIDENT] {incident}")
    app = build_graph()
    app.invoke({"incident": incident})


if __name__ == "__main__":
    main()
