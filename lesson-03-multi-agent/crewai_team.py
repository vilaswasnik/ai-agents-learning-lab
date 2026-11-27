"""
LESSON 03a — CrewAI: agents as a TEAM with job titles.

CrewAI's idea: instead of one agent doing everything, define a CREW —
agents with a role, a goal and a backstory — and give the crew TASKS.
The crew runs the tasks in order, each agent building on the last one's
output. Think of it as hiring a small team:

    Investigator  -> finds the facts
    Planner       -> writes the fix plan
    Reviewer      -> checks the plan for risk (our human-approval idea,
                     done by an agent first — the human still approves
                     anything real, in lesson 02's graph)

Run options:
    python lesson-03-multi-agent/crewai_team.py --simulate
        Walks the same three roles in plain Python with the mock brain.
        No extra packages, no API key — watch WHO produces WHAT.

    python lesson-03-multi-agent/crewai_team.py
        The real CrewAI version. Needs: pip install crewai
        and a real LLM provider in .env (LLM_PROVIDER=openai etc.).
        CrewAI talks to the provider through the same settings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

INCIDENT = "web01 disk is full, the website is very slow"
FACTS = ("disk / = 97% full; /var/log = 40 GB (httpd logs not rotated for "
         "21 days); httpd running; load normal")


def simulate() -> None:
    """The CrewAI pattern, acted out with the shared mock brain."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from shared.llm import get_chat_model

    model = get_chat_model()

    def run_agent(role: str, task_prompt: str) -> str:
        messages = [
            SystemMessage(content=f"You are the {role} on a Linux operations team."),
            HumanMessage(content=task_prompt),
        ]
        return str(model.invoke(messages).content)

    print(f"[SIMULATE — CrewAI pattern]\n[INCIDENT] {INCIDENT}\n")

    investigation = run_agent(
        "Investigator",
        f"TASK: classify\nIncident report: {INCIDENT}\nFacts available: {FACTS}\n"
        "Reply with TYPE=<type> and SEVERITY=<level>.",
    )
    print(f"--- Investigator (find facts & classify) ---\n{investigation}\nFacts: {FACTS}\n")

    plan = run_agent(
        "Planner",
        f"TASK: plan\nIncident: {INCIDENT}\nFacts: {FACTS}\n"
        "Write a numbered fix plan, one step per line.",
    )
    print(f"--- Planner (write the fix plan) ---\n{plan}\n")

    review = (
        "Reviewer notes: plan is read-only until step 2; log rotation is "
        "low-risk and reversible; no service restart needed; recommend the "
        "human operator approves. (In simulate mode this review is canned — "
        "with a real LLM the Reviewer agent argues about the actual plan.)"
    )
    print(f"--- Reviewer (challenge the plan) ---\n{review}\n")
    print("Crew done. Notice the shape: role -> task -> output handed to the next role.")


def real_crew() -> None:
    from crewai import Agent, Crew, Task

    investigator = Agent(
        role="Investigator",
        goal="Find the facts of an incident and classify it",
        backstory="A senior Linux administrator who never guesses — "
                  "she checks first and reports only what she verified.",
    )
    planner = Agent(
        role="Planner",
        goal="Turn incident facts into a safe, step-by-step fix plan",
        backstory="An SRE who writes plans a junior could follow at 3 AM, "
                  "with a backout step always included.",
    )
    reviewer = Agent(
        role="Reviewer",
        goal="Challenge fix plans: risk, downtime, backout",
        backstory="A change-manager who has seen every way a 'simple fix' "
                  "takes production down, and asks about those first.",
    )

    t1 = Task(description=f"Classify this incident and list the verified facts: {INCIDENT}. Known facts: {FACTS}",
              expected_output="Incident type, severity, and the fact list.",
              agent=investigator)
    t2 = Task(description="Write a numbered fix plan from the Investigator's output.",
              expected_output="A step-by-step fix plan with a backout step.",
              agent=planner)
    t3 = Task(description="Review the Planner's plan: call out risks, downtime and the backout. End with APPROVE or REVISE.",
              expected_output="Risk review ending in APPROVE or REVISE.",
              agent=reviewer)

    crew = Crew(agents=[investigator, planner, reviewer], tasks=[t1, t2, t3], verbose=True)
    print(crew.kickoff())


if __name__ == "__main__":
    if "--simulate" in sys.argv:
        simulate()
    else:
        real_crew()
