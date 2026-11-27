"""
LESSON 01 — LangChain basics: prompt -> model -> answer.

LangChain's core pattern is a CHAIN: pieces connected with the pipe | .

    prompt | model | output_parser

- ChatPromptTemplate : your instructions, with {placeholders} filled at run time
- model              : ANY chat model from shared/llm.py (mock/OpenAI/Azure/Bedrock)
- StrOutputParser    : turns the model's message into a plain string

Run it:
    python lesson-01-langchain-basics/chat_chain.py
    python lesson-01-langchain-basics/chat_chain.py "Explain what Ansible does"

Then edit .env, set LLM_PROVIDER=openai (and your key), run again —
NOTICE: this file does not change at all. That is the whole point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from shared.llm import get_chat_model, provider_name


def main() -> None:
    question = " ".join(sys.argv[1:]) or (
        "TASK: explain — in 3 short lines, what is an AI agent? "
        "Answer like you are teaching a Linux system administrator."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a patient teacher for a senior Linux & DevOps engineer who is learning AI agents. Use simple words and short lines."),
            ("human", "{question}"),
        ]
    )

    model = get_chat_model()

    # THE chain — read it left to right: fill the prompt, ask the model,
    # give me back just the text.
    chain = prompt | model | StrOutputParser()

    print(f"[provider: {provider_name()}]\n")
    answer = chain.invoke({"question": question})
    print(answer)


if __name__ == "__main__":
    main()
