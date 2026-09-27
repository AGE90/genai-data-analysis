"""LLM-as-judge tasks: grade another model's output against the conversation it came from.

Fields put `reasoning` first so the model explains before deciding. The bool fields also map
to Jev Noul questions, so a System One model can act as a cheap judge (backends.jev).
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from genaianalysis.schema import Conversation, Message, Task

CANDIDATE = "=== ANALYSIS TO EVALUATE ==="


class ArcJudgment(BaseModel):
    reasoning: str = Field(description="Step-by-step check of the analysis against the chat")
    initial_label_supported: bool = Field(
        description="The analysis' initial sentiment label matches how the customer comes across "
        "at the beginning of the conversation"
    )
    final_label_supported: bool = Field(
        description="The analysis' final sentiment label matches how the customer comes across "
        "at the end of the conversation"
    )
    justifications_faithful: bool = Field(
        description="Both justifications describe this conversation and state nothing that the "
        "conversation does not show"
    )
    score: Literal[1, 2, 3, 4, 5] = Field(description="Overall quality, 5 = fully correct")


ARC_JUDGE = Task(
    name="arc_judge",
    output_type=ArcJudgment,
    instructions=(
        "You are a strict reviewer of sentiment analyses of customer-service chats. The chat is "
        f"followed by a line '{CANDIDATE}' and a JSON analysis produced by another system. Check "
        "each label and each justification against the chat only. Labels are positive, negative "
        "or neutral; neutral means no clear positive or negative emotion (plain questions, "
        "information exchange, flat acknowledgement). Do not reward length or confident tone."
    ),
)


class PairwiseChoice(BaseModel):
    reasoning: str = Field(description="Compare both options against the chat before choosing")
    choice: Literal["A", "B"] = Field(description="The option that better answers the question")


FINAL_SENTIMENT_PAIRWISE = Task(
    name="final_sentiment_pairwise",
    output_type=PairwiseChoice,
    instructions=(
        "You are an impartial judge. After the chat you get a question and two options, A and "
        "B. Pick the option that is better supported by the chat. Option order is random and "
        "carries no information. Labels are positive, negative or neutral; neutral means no clear "
        "positive or negative emotion (plain questions, information exchange, flat "
        "acknowledgement)."
    ),
)


def with_candidate(conv: Conversation, candidate: dict[str, Any], suffix: str) -> Conversation:
    """The chat plus a trailing message holding the output to grade (for ARC_JUDGE)."""
    text = json.dumps(candidate, ensure_ascii=False, indent=1)
    return conv.model_copy(
        update={
            "id": f"{conv.id}:{suffix}",
            "messages": [*conv.messages, Message(sender=CANDIDATE, text=text)],
        }
    )


def with_options(conv: Conversation, question: str, a: str, b: str, suffix: str) -> Conversation:
    """The chat plus the question and options A/B (for pairwise tasks)."""
    text = f"Question: {question}\nA: {a}\nB: {b}"
    return conv.model_copy(
        update={
            "id": f"{conv.id}:{suffix}",
            "messages": [*conv.messages, Message(sender="=== QUESTION ===", text=text)],
        }
    )
