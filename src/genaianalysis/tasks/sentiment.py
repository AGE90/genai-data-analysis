"""Sentiment tasks. SENTIMENT_ARC is the port of the legacy WhatsApp customer-service analysis."""

from typing import Literal

from pydantic import BaseModel, Field

from genaianalysis.schema import Task

Label = Literal["positive", "negative", "neutral"]


class MessageSentiment(BaseModel):
    sentiment: Label = Field(
        description="The overall sentiment expressed by the author of the text"
    )


class SentimentArc(BaseModel):
    initial_sentiment: Label = Field(
        description="The customer's sentiment at the beginning of the conversation"
    )
    initial_sentiment_justification: str = Field(
        description="Brief justification of the initial sentiment"
    )
    final_sentiment: Label = Field(
        description="The customer's sentiment at the end of the conversation"
    )
    final_sentiment_justification: str = Field(
        description="Brief justification of the final sentiment"
    )


MESSAGE_SENTIMENT = Task(
    name="message_sentiment",
    output_type=MessageSentiment,
    instructions=(
        "You are an expert in sentiment analysis. Classify the sentiment of the text as positive, "
        "negative or neutral. Texts may be informal social-media posts with slang, sarcasm, emojis "
        "and @user mentions; judge the author's attitude, not the topic."
    ),
)

SENTIMENT_ARC = Task(
    name="sentiment_arc",
    output_type=SentimentArc,
    instructions=(
        "You are an expert in sentiment analysis of customer-service chats. Each line is "
        "'<timestamp> - <sender>: <message>'. Determine the customer's sentiment at the beginning "
        "and at the end of the conversation (positive, negative or neutral) and justify each one "
        "briefly, reflecting the customer's expressed emotions or issues. Write the justifications "
        "in the language of the conversation."
    ),
)
