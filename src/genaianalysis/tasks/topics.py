"""Name a cluster of conversations (used by topics.label_clusters)."""

from pydantic import BaseModel, Field

from genaianalysis.schema import Task


class TopicLabel(BaseModel):
    name: str = Field(description="Short topic name, 2-5 words")
    description: str = Field(description="One sentence describing what these conversations share")


TOPIC_LABEL = Task(
    name="topic_label",
    output_type=TopicLabel,
    instructions=(
        "You receive sample conversations that were clustered together by meaning. Name the "
        "common topic (what the conversations are about, not their tone) and describe it in one "
        "sentence. Answer in the language of the conversations."
    ),
)
