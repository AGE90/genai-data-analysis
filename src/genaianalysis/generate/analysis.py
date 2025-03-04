"""
This module contains the code to generate a response from the Google Vertex AI API.
"""
from google import genai
from google.genai import types

from genaianalysis.credentials import vertex_app_creds


def generate():
    """
    Generate response
    """
    client = genai.Client(
        vertexai=True,
        project="erudite-realm-451722-v1",
        location="us-central1",
        credentials=vertex_app_creds
    )

    model = "gemini-2.0-flash-001"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Explain quantum computing""")
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")


generate()
