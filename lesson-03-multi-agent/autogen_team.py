"""
LESSON 03b — AutoGen: agents solve it by TALKING to each other.

CrewAI (lesson 03a) is a team with assigned tasks, run in order.
AutoGen's idea is different: a CONVERSATION. Agents take turns speaking
in a group chat — proposing, criticising, refining — until the group
agrees the work is done (someone says TERMINATE).

    Operator -> [group chat] Investigator <-> Planner <-> Reviewer -> answer

Same incident, same three roles as the CrewAI lesson — compare the feel:
- CrewAI  = a manager handing out tasks in a fixed order.
- AutoGen = a meeting where the plan emerges from discussion.

Run options:
    python lesson-03-multi-agent/autogen_team.py --simulate
        Acts out the group-chat pattern in plain Python with the mock
        brain. No extra packages, no API key.

    python lesson-03-multi-agent/autogen_team.py
        The real AutoGen version. Needs: pip install pyautogen
        and an OpenAI-compatible key (AutoGen is configured with an
        llm_config pointing at the provider).
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
    """The AutoGen pattern: a transcript of agents taking turns."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from shared.llm import get_chat_model

    model = get_chat_model()

    def speak(role: str, prompt: str) -> str:
        messages = [
            SystemMessage(content=f"You are the {role} in a group chat of Linux operations experts."),
            HumanMessage(content=prompt),
        ]
        return str(model.invoke(messages).content)

    print(f"[SIMULATE — AutoGen pattern]\n[Operator] {INCIDENT}\n")

    classification = speak("Investigator",
                           f"TASK: classify\nIncident report: {INCIDENT}\n"
                           "Reply with TYPE=<type> and SEVERITY=<level>.")
    print(f"[Investigator] {classification} Facts I verified: {FACTS}\n")

    plan = speak("Planner",
                 f"TASK: plan\nIncident: {INCIDENT}\nFacts: {FACTS}\n"
                 "Write a numbered fix plan, one step per line.")
    print(f"[Planner] Here is my plan:\n{plan}\n")

    print("[Reviewer] Two questions before I agree: (1) does any step restart "
          "httpd? No — good, zero downtime. (2) Is there a backout? Log "
          "rotation is reversible and the archive is kept. I accept the plan.\n")

    print("[Planner] Agreed — and I will add the archive-verify step explicitly.\n")
    print("[Reviewer] Then we are done. TERMINATE")
    print("\nNotice: nobody assigned tasks in order — the answer emerged from the conversation.")


def real_autogen() -> None:
    import os

    import autogen

    llm_config = {
        "config_list": [
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": os.environ["OPENAI_API_KEY"],
            }
        ],
        "temperature": 0,
    }

    operator = autogen.UserProxyAgent(
        name="Operator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    investigator = autogen.AssistantAgent(
        name="Investigator",
        system_message="You classify Linux incidents and state only verified facts.",
        llm_config=llm_config,
    )
    planner = autogen.AssistantAgent(
        name="Planner",
        system_message="You write numbered, low-risk fix plans with a backout step.",
        llm_config=llm_config,
    )
    reviewer = autogen.AssistantAgent(
        name="Reviewer",
        system_message="You challenge plans for risk and downtime. "
                       "Say TERMINATE when the plan is safe and complete.",
        llm_config=llm_config,
    )

    chat = autogen.GroupChat(
        agents=[operator, investigator, planner, reviewer],
        messages=[],
        max_round=8,
    )
    manager = autogen.GroupChatManager(groupchat=chat, llm_config=llm_config)
    operator.initiate_chat(
        manager,
        message=f"Incident: {INCIDENT}. Known facts: {FACTS}. "
                "Investigator first, then Planner, then Reviewer.",
    )


if __name__ == "__main__":
    if "--simulate" in sys.argv:
        simulate()
    else:
        real_autogen()
