"""
shared/llm.py — ONE place that decides which LLM "brain" every lesson uses.

This is the most important idea in the whole lab:

    Your agent code should NEVER hard-code a provider.
    It asks this factory for "a chat model", and the .env file decides
    whether that is OpenAI, Azure OpenAI, AWS Bedrock — or the built-in
    MOCK brain that needs no API key at all.

Switch providers by changing ONE line in .env:

    LLM_PROVIDER=mock          <- default, no key, runs anywhere
    LLM_PROVIDER=openai        <- needs OPENAI_API_KEY
    LLM_PROVIDER=azure_openai  <- needs AZURE_OPENAI_* settings
    LLM_PROVIDER=bedrock       <- needs AWS credentials + BEDROCK_MODEL_ID

Every lesson calls get_chat_model() and then uses the model the same way:
    model.invoke(messages)  ->  AIMessage with .content
That sameness is exactly what LangChain gives you.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# --------------------------------------------------------------------------
# The MOCK brain.
#
# A real chat model is just: messages in -> one message out.
# This mock subclasses LangChain's BaseChatModel (the same base class the
# real OpenAI/Azure/Bedrock models use) and answers with simple keyword
# rules, so every lesson runs end-to-end with NO API key.
#
# The lessons mark what they want in the prompt, e.g. "TASK: classify".
# The mock looks for that marker plus incident keywords (disk/cpu/service)
# and returns a canned, sensible answer. With a real LLM the same prompts
# are answered by the model itself — the lesson code does not change.
# --------------------------------------------------------------------------

def _mock_reply(prompt: str) -> str:
    text = prompt.lower()

    incident_type = "general"
    if "disk" in text or "/var" in text or "filesystem" in text:
        incident_type = "disk"
    elif "cpu" in text or "load average" in text:
        incident_type = "cpu"
    elif "service" in text or "httpd" in text or "apache" in text or "down" in text:
        incident_type = "service"
    elif "memory" in text or "ram" in text or "oom" in text:
        incident_type = "memory"

    if "task: classify" in text:
        severity = "HIGH" if incident_type in ("disk", "service") else "MEDIUM"
        return f"TYPE={incident_type}\nSEVERITY={severity}"

    if "task: plan" in text:
        plans = {
            "disk": (
                "1. Archive logs older than 7 days from /var/log to backup storage\n"
                "2. Run log rotation (logrotate) for httpd and system logs\n"
                "3. Verify no application is holding deleted-but-open files\n"
                "4. Re-check filesystem usage"
            ),
            "cpu": (
                "1. Identify the top CPU-consuming process\n"
                "2. Check whether it is a legitimate workload or a stuck process\n"
                "3. If stuck, restart only that process's service\n"
                "4. Re-check load average for 5 minutes"
            ),
            "service": (
                "1. Check the service status and the last 50 log lines\n"
                "2. Check the port is free and config test passes\n"
                "3. Restart the service\n"
                "4. Verify the service responds on its port"
            ),
            "memory": (
                "1. Identify the top memory-consuming process\n"
                "2. Check for a memory leak pattern (steady growth)\n"
                "3. Restart the leaking service in a controlled way\n"
                "4. Re-check free memory"
            ),
            "general": (
                "1. Collect basic system facts (uptime, disk, memory, top processes)\n"
                "2. Review recent system logs for errors\n"
                "3. Apply the smallest safe corrective action\n"
                "4. Re-check system health"
            ),
        }
        return plans[incident_type]

    if "task: report" in text:
        return (
            "Incident handled. The problem was classified, facts were checked, "
            "a fix plan was approved by the operator, the steps were executed, "
            "and the system was re-checked afterwards to confirm the fix worked."
        )

    if "task: explain" in text:
        return (
            "An AI agent is a program that uses an LLM to decide what to do "
            "next: it reads the situation, picks a tool or action, looks at "
            "the result, and repeats until the goal is met — with a human "
            "approving anything risky."
        )

    return (
        "I am the MOCK brain (no API key). I understood your message and "
        "would normally answer it with a real LLM. Set LLM_PROVIDER in .env "
        "to openai, azure_openai or bedrock to go live."
    )


class MockChatModel(BaseChatModel):
    """A LangChain chat model with canned, rule-based answers."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        reply = _mock_reply(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    @property
    def _llm_type(self) -> str:
        return "mock"


# --------------------------------------------------------------------------
# The factory — the ONLY function the lessons call.
# Real providers are imported lazily, so you only need the packages and
# keys for the provider you actually select.
# --------------------------------------------------------------------------

def get_chat_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()

    if provider == "mock":
        return MockChatModel()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=0,
        )

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model=os.environ["BEDROCK_MODEL_ID"],
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            temperature=0,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Use one of: mock, openai, azure_openai, bedrock."
    )


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "mock").strip().lower()
