"""
Wires the three grounding tools (metrics calculator, nutrient lookup, and
guideline retriever) into a single LangChain agent, plus conversation memory.

Design note: all three tools exist to keep the LLM from inventing numbers.
- calculate_profile_metrics -> deterministic math (BMI, TDEE, sleep deficit)
- lookup_food_nutrients     -> exact table lookup (calories/macros per food)
- retrieve_nutrition_guidelines -> hybrid search over guideline documents

The LLM's job is narrow by design: interpret the user's question, call the
right tool(s), and weave the results into personalized, cited prose. It
should not be computing BMI or recalling nutrient facts on its own.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from metrics_tool import build_metrics_tool
from nutrient_lookup import build_nutrient_lookup_tool
from retriever import build_hybrid_retriever

load_dotenv()

SYSTEM_PROMPT = """You are a nutrition and wellness assistant. You help users \
understand their nutrition, calorie, and sleep needs based on their personal \
profile and evidence-based guidelines.

Rules you must follow:
1. For any BMI, calorie, or sleep-deficit calculation, ALWAYS call \
calculate_profile_metrics. Never compute or estimate these yourself.
2. For any specific food's calorie or macronutrient content, ALWAYS call \
lookup_food_nutrients. If it returns no match, tell the user the food isn't \
in the database rather than guessing a value.
3. For general nutrition or sleep guidance, ALWAYS call \
retrieve_nutrition_guidelines and cite the source document in your answer.
4. You are not a doctor or registered dietitian. Frame your answers as \
educational information, not medical advice. If a user's profile suggests \
something concerning (e.g. very low reported sleep, very high or low BMI), \
say so plainly and suggest they speak with a healthcare professional — do \
not just give generic advice as if nothing unusual was reported.
5. Be concise and specific. Ground every claim in a tool result.
"""


def build_retriever_tool():
    retriever = build_hybrid_retriever()

    class RetrieverInput(BaseModel):
        query: str = Field(description="A specific nutrition or sleep guideline question")

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant guideline content found for this query."

        formatted = []
        for doc in docs:
            source = Path(doc.metadata.get("source", "unknown")).name
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    return StructuredTool.from_function(
        func=_run,
        name="retrieve_nutrition_guidelines",
        description=(
            "Retrieves relevant passages from nutrition and sleep guideline "
            "documents (Dietary Guidelines for Americans, WHO fact sheets, CDC "
            "sleep guidance). Always use this for general nutrition/sleep advice "
            "questions and cite the returned source in your final answer."
        ),
        args_schema=RetrieverInput,
    )


def build_agent_executor(verbose: bool = False) -> AgentExecutor:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [
        build_metrics_tool(),
        build_nutrient_lookup_tool(),
        build_retriever_tool(),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
    )


if __name__ == "__main__":
    executor = build_agent_executor(verbose=True)

    sample_question = (
        "I'm a 30 year old male, 80kg, 178cm, moderately active, and I get "
        "about 6 hours of sleep a night. I usually eat 2 eggs and a cup of "
        "brown rice for breakfast — is that a good start to my day, and what "
        "should I change?"
    )
    response = executor.invoke({"input": sample_question})
    print("\n=== RESPONSE ===\n")
    print(response["output"])
