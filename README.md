# AI Agents Learning Lab

Learn the agent technologies by running ONE familiar scenario four ways:
a **Linux server incident triage agent** (disk full / high CPU / service
down) — detect → classify → check facts → plan → **human approval** →
execute → verify → report. The same flow as real operations work.

## The technologies, in learning order

| Lesson | Technology | The idea in one line |
|---|---|---|
| 01 | **LangChain** | The plumbing: prompt → model → answer, as a chain |
| 02 | **LangGraph** | The flow chart: steps, decisions and a human-approval stop, as a state machine |
| 03a | **CrewAI** | A team: agents with job titles dividing the work |
| 03b | **AutoGen** | A meeting: agents solve it by talking to each other |
| 04 | **Copilot Studio** | The no-code version, living inside Microsoft 365 / Teams |

The **LLM platforms** — OpenAI, Azure OpenAI, AWS Bedrock — are the
interchangeable "brain". Every lesson gets its model from
`shared/llm.py`, and `.env` decides which brain is plugged in.
Change one line, re-run, and the lesson code is identical.

## Quick start (no API key needed)

```bash
cd ai-agents-learning-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # LLM_PROVIDER=mock — the built-in fake brain

python lesson-01-langchain-basics/chat_chain.py
python lesson-02-langgraph-agent/triage_agent.py "db01 high CPU, load average 40"
python lesson-03-multi-agent/crewai_team.py --simulate
python lesson-03-multi-agent/autogen_team.py --simulate
```

The mock brain answers with simple keyword rules (see `shared/llm.py`),
so everything runs end-to-end offline. When you are ready, put a real
key in `.env`, set `LLM_PROVIDER=openai` (or `azure_openai` / `bedrock`),
and run the same files again.

## The LangGraph flow (lesson 02)

```
START -> classify -> check -> plan -> approval --YES--> execute -> verify -> report -> END
                                          |
                                          +-------NO-----> report -> END
```

## Repo map

```
shared/llm.py                  the provider factory + the mock brain
lesson-01-langchain-basics/    chat_chain.py
lesson-02-langgraph-agent/     triage_agent.py
lesson-03-multi-agent/         crewai_team.py, autogen_team.py
lesson-04-copilot-studio/      README.md (step-by-step build guide)
```
