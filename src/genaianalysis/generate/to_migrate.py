Make_dataset.py
"""
This module provides the required classes and
functions to create and process the dataset for the
Wolkvox chat data processing project.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union
import mimetypes

import pandas as pd
import pandas_gbq
from bs4 import BeautifulSoup
from google.cloud import firestore, storage
from google.cloud.firestore import FieldFilter, Or

import wxprocessing.utils.paths as path
from wxprocessing.credentials import (
    firestore_credentials,
    storage_credentials,
    gbq_credentials,
    gbq_project_id,
)
from wxprocessing.utils.read_sql import read_sql_file

pandas_gbq.context.project = gbq_project_id
pandas_gbq.context.credentials = gbq_credentials

logger = logging.getLogger(__name__)

def extract_media_from_json(
    json_media: Dict,
) -> Optional[Tuple[str, str, str]]:
    """
    Extracts multimedia data from a JSON object containing HTML content.

    This function parses the HTML content stored in a JSON object to find the
    specified type of multimedia (audio, image, video, document) and extracts
    its MIME type and base64 encoded data.

    Parameters
    ----------
    json_media : Dict
        A dictionary containing HTML content with embedded multimedia data.
        The HTML content should be stored under the key "data".

    Returns
    -------
    Optional[Tuple[str, str, str]]:
        A tuple containing the media type, MIME type and base64 encoded data of the
        extracted multimedia content. If no multimedia is found, returns None.

    Raises
    ------
    ValueError
        If the provided JSON does not contain HTML data.
    """
    # Generate HTML soup
    # print(json_media) 
    # print(json_media.get("data", ""))
    html_data = json_media.get("data", "")
    
    if not html_data:
        raise ValueError("No HTML data found in the provided JSON.")

    # Generate HTML soup
    soup = BeautifulSoup(json_media.get("data", ""), "html.parser")
    # print(soup.prettify())

    extracted_data = []
    seen_base64 = set()  # Track unique base64 strings to avoid duplication

    # Find all tags asociated with media
    for tag in soup.find_all(["audio", "img", "video", "a"]):
        media_type = None
        mime_type = None
        base64_data = None

        # Check for <a> tag (usually for documents or downloadable media)
        if tag.name == "a" and (href := tag.get("href", "")).startswith("data:"):
            try:
                # Remove the prefix, extract MIME type and base64 data
                mime_type, base64_data = href.removeprefix("data:").split(";base64,", 1)
                # Assuming <a> tags of type "application" are documents
                media_type = (
                    "doc"
                    if mime_type.split("/", 1)[0] in ["text", "document", "application"]
                    else mime_type.split("/", 1)[0]
                )
            except ValueError:
                continue  # Skip if href is malformed

        # Check for <img>, <audio>, <video> tags
        elif tag.name in ("img", "audio", "video"):
            source_tag = tag.find("source", src=True)
            src = (source_tag or tag).get("src", "")
            if src.startswith("data:"):
                try:
                    # Remove the prefix extract MIME type and base64 data
                    mime_type, base64_data = src.removeprefix("data:").split(
                        ";base64,", 1
                    )
                    media_type = mime_type.split("/", 1)[0]
                except ValueError:
                    continue  # Skip if src is malforme

        # If base64_data is found and not already seen, add to the list
        if base64_data and base64_data not in seen_base64:
            extracted_data.append((media_type, mime_type, base64_data))
            seen_base64.add(base64_data)

    return extracted_data

def get_media_blobs_ref(
    conversations: List[Dict],
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Extracts and returns lists of references to multimedia files from chat conversations.

    This function processes a list of chat conversations, extracts references to audio,
    image, video, and document files, and organizes them into separate lists.

    Parameters
    ----------
    conversations : List[Dict]
        A list of dictionaries representing chat conversations. Each dictionary should
        contain a 'conversation' key, where 'conversation' is a list of message dictionaries.

    Returns
    -------
    Tuple[List[str], List[str], List[str], List[str]]
        A tuple containing four lists:
        - List of audio file references
        - List of image file references
        - List of video file references
        - List of document file references
    """
    audios = []
    images = []
    videos = []
    docs = []
    raw_json = []

    for conversation in conversations:
        for msg in conversation.get("conversation", []):
            message_content = msg.get("message", "")
            if "audio_msg" in message_content:
                audios.append(message_content)
            elif "image_msg" in message_content:
                images.append(message_content)
            elif "video_msg" in message_content:
                videos.append(message_content)
            elif "doc_msg" in message_content:
                docs.append(message_content)
            elif "raw_media_msg" in message_content:
                raw_json.append(message_content)

    return audios, images, videos, docs, raw_json

def get_media_blobs_ref_json(
    conversations: List[Dict],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Extracts and returns lists of references to multimedia files from chat conversations.

    This function processes a list of chat conversations, extracts references to audio,
    image, video, and document files, and organizes them into separate lists.

    Parameters
    ----------
    conversations : List[Dict]
        A list of dictionaries representing chat conversations. Each dictionary should
        contain a 'conn_id' key and a 'conversation' key, where 'conversation' is a list
        of message dictionaries.

    Returns
    -------
    Tuple[List[str], List[str], List[str], List[str]]
        A tuple containing four lists:
        - List of audio file references
        - List of image file references
        - List of video file references
        - List of document file references
    """
    audios = []
    images = []
    videos = []
    docs = []

    for conversation in conversations:
        conn_id = conversation.get("conn_id")
        for msg in conversation.get("conversation", []):
            message_content = msg.get("message", "")
            if "audio_msg" in message_content:
                audios.append(f"{conn_id}/{message_content}.json")
            elif "image_msg" in message_content:
                images.append(f"{conn_id}/{message_content}.json")
            elif "video_msg" in message_content:
                videos.append(f"{conn_id}/{message_content}.json")
            elif "doc_msg" in message_content:
                docs.append(f"{conn_id}/{message_content}.json")

    return audios, images, videos, docs

class MakeDataset:
    """
    A class to handle dataset creation and processing.
    """

    def get_processed_media(
        self,
        conversation_date: str,
        sql_file: str = "processed_media.sql",
    ) -> pd.DataFrame:
        """
        Retrieves processed media data for a given conversation date by executing a
        specified SQL query.

        Parameters
        ----------
        conversation_date : str
            The date of the conversation in 'YYYY-MM-DD' format.
        sql_file : str, optional
            The name of the SQL file containing the query, by default "processed_media.sql".

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the processed media data.
        """

        slq_dir = path.sql_dir(sql_file)
        sql = read_sql_file(slq_dir).format(conversation_date=f"'{conversation_date}'")

        df = pandas_gbq.read_gbq(
            query_or_table=sql,
            progress_bar_type=None,
        )

        df["media_ref"] = df["media_ref"].astype(object)
        return df

    def get_media_text(
        self,
        conversation_date: str,
        sql_file: str = "media_text.sql",
    ) -> pd.DataFrame:
        """
        Retrieve media-related text data for a specific conversation date.

        Parameters
        ----------
        conversation_date : str
            The date of the conversation for which to retrieve media text data.
            The date should be in the format 'YYYY-MM-DD'.
        sql_file : str, optional
            The name of the SQL file containing the query,
            by default "media_text.sql".

        Returns
        -------
        pd.DataFrame
            A DataFrame containing media-related text data.
        """
        slq_dir = path.sql_dir(sql_file)
        sql = read_sql_file(slq_dir).format(conversation_date=f"'{conversation_date}'")

        df = pandas_gbq.read_gbq(
            query_or_table=sql,
            progress_bar_type=None,
        )

        dtypes = {
            "media_ref_path": str,
            "media_type": str,
            "generated_text": str,
            "confidence": float,
            "task": str,
        }
        df = df.astype(dtypes)
        df["generation_date"] = pd.to_datetime(df["generation_date"])
        df["conversation_date"] = pd.to_datetime(df["conversation_date"])

        # Split media_ref
        df[["conn_id", "media_ref"]] = df["media_ref_path"].str.split(
            "/", n=1, expand=True
        )
        df["media_ref"] = df["media_ref"].str.replace(".json", "")
        df["message_index"] = df["media_ref"].str.extract("(\d+)").astype(int)

        return df

    def get_low_confidence_transcriptions(
        self,
        conversation_date: str,
        sql_file: str = "low_confidence_transcriptions.sql",
    ) -> pd.DataFrame:
        """
        Retrieve audio transcriptions with low confidence for a specific conversation date.

        Parameters
        ----------
        conversation_date : str
            The date of the conversation for which to retrieve media text data.
            The date should be in the format 'YYYY-MM-DD'.
        sql_file : str, optional
            The name of the SQL file containing the query,
            by default "low_confidence_transcriptions.sql".

        Returns
        -------
        pd.DataFrame
            A DataFrame containing media-related text data.
        """
        slq_dir = path.sql_dir(sql_file)
        sql = read_sql_file(slq_dir).format(conversation_date=f"'{conversation_date}'")

        df = pandas_gbq.read_gbq(
            query_or_table=sql,
            progress_bar_type=None,
        )

        dtypes = {
            "media_ref": str,
            "media_type": str,
            "mime_type": str,
            "generated_text": str,
            "confidence": float,
            "model": str,
        }
        df["generation_date"] = pd.to_datetime(df["generation_date"])
        df["conversation_date"] = pd.to_datetime(df["conversation_date"])
        df = df.astype(dtypes)

        return df

class GetFirestoreData:
    """
    A class to retrieve data from Firestore.

    This class provides methods to query chat conversations from a Firestore
    database that contain multimedia content on a specific date.

    Attributes
    ----------
    database_name : str
        The name of the Firestore database.
    db : firestore.Client
        The Firestore client used to interact with the Firestore service.
    """

    def __init__(
        self,
        database_name: str = "wx-chats",
    ) -> None:
        """
        Initializes the GetFirestoreData class with the specified database name.

        Parameters
        ----------
        database_name : str, optional
            The name of the Firestore database. Default is "wx-chats".
        """
        self.database_name = database_name
        # Initialize Firestore database
        self.db = firestore.Client(
            credentials=firestore_credentials,
            database=database_name,
        )

        # Create a referencce to the conversations collection
        self.conversations_ref = self.db.collection("conversations")

    def query_conversations_by_date(
        self,
        date: str,
    ) -> List[Dict]:
        """
        Query Firestore to retrieve all chat conversations on a specific date.

        Parameters
        ----------
        date : str
            The date to filter the conversations by, in the format 'YYYY-MM-DD'.

        Returns
        -------
        List[Dict]
            A list of dictionaries, where each dictionary represents a
            chat conversation with its details and messages.
        """
        # Create filter for date
        filter_date = FieldFilter("conversation_date", "==", date)

        # Execute the query
        conversations = self.conversations_ref.where(filter=filter_date).stream(
            timeout=480
        )

        # Build list with conversations
        data = []
        for conversation in conversations:
            item = conversation.to_dict()
            msgs = []
            for messages_ref in conversation.reference.collections():
                messages = messages_ref.stream()
                for message in messages:
                    msgs.append(message.to_dict())
            item["conversation"] = msgs
            data.append(item)

        return data

    def query_conversations_with_media_by_date(
        self,
        date: str,
    ) -> List[Dict]:
        """
        Query Firestore to retrieve chat conversations with multimedia content on a specific date.

        Parameters
        ----------
        date : str
            The date to filter the conversations by, in the format 'YYYY-MM-DD'.

        Returns
        -------
        List[Dict]
            A list of dictionaries, where each dictionary represents a
            chat conversation with its details and messages.
        """
        # Create filter for date
        filter_date = FieldFilter("conversation_date", "==", date)

        # Create the union filter of the media filters (queries)
        or_filter = Or(
            filters=[
                FieldFilter("n_audios", ">", 0),
                FieldFilter("n_images", ">", 0),
                FieldFilter("n_videos", ">", 0),
                FieldFilter("n_docs", ">", 0),
            ]
        )

        # Execute the query
        conversations = (
            self.conversations_ref.where(filter=filter_date)
            .where(filter=or_filter)
            .stream(timeout=480)
        )

        # Build list with conversations
        data = []
        for conversation in conversations:
            item = conversation.to_dict()
            msgs = []
            for messages_ref in conversation.reference.collections():
                messages = messages_ref.stream()
                for message in messages:
                    msgs.append(message.to_dict())
            item["conversation"] = msgs
            data.append(item)

        return data

    def delete_conversations_before_date(
        self,
        date: str,
    ) -> None:
        """
        Delete all conversations before a specific date.

        Parameters
        ----------
        date : str
            The date to filter the conversations by, in the format 'YYYY-MM-DD'.
        """
        # Create filter for date
        filter_date = FieldFilter("conversation_date", "<", date)

        # Execute the query
        conversations = self.conversations_ref.where(filter=filter_date).stream(
            timeout=480
        )

        for conversation in conversations:
            for messages_ref in conversation.reference.collections():
                messages = messages_ref.stream()
                for message in messages:
                    # Delete the messages
                    message.reference.delete()
            # Delete the conversation
            conversation.reference.delete()

class GetCloudStorageData:
    """
    A class to retrieve data from Google Cloud Storage.

    This class provides methods to list all blobs in a specified bucket and
    to retrieve JSON data from a specific blob.

    Attributes
    ----------
    bucket_name : str
        The name of the Cloud Storage bucket.
    client : storage.Client
        The Cloud Storage client used to interact with the storage service.
    bucket : storage.Bucket
        The Cloud Storage bucket object.
    """

    def __init__(
        self,
        bucket_name: str = "wx-chats-media",
    ) -> None:
        """
        Initializes the GetCloudStorageData class with the specified bucket
        name.

        Parameters
        ----------
        bucket_name : str, optional
            The name of the Cloud Storage bucket. Default is "wx-chats-media".
        """
        self.bucket_name = bucket_name
        self.client = storage.Client(credentials=storage_credentials)
        self.bucket = self.client.get_bucket(bucket_name)

    def get_blobs_list(self, max_results: int = None) -> List[str]:
        """
        Retrieves a list of all blob names in the specified Cloud Storage
        bucket.

        Parameters
        ----------
        max_results : int, optional
            The maximum number of blobs to retrieve. If None, all blobs
            will be retrieved.

        Returns
        -------
        List[str]
            A list of blob names (as strings) present in the specified
            bucket.
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, max_results=max_results)
            return [blob.name for blob in blobs]

        except Exception as e:
            logger.error("An error occurred while listing blobs: %s", e)
            return None

    def get_media_data(
        self,
        blob_path: str,
        json_type: bool = False,
    ) -> Union[Dict, bytes]:  # Update return type hint
        """
        Retrieves data from a specified blob in the Cloud Storage bucket.
        If the data is a JSON file it must be specified with the json_type
        parameter.

        Parameters
        ----------
        blob_path : str
            The path to the blob within the bucket.

        Returns
        -------
        Union[Dict, bytes]
            The JSON data retrieved from the blob as a dictionary,
            the raw bytes if not JSON, or None if an error occurs.
        """
        try:
            blob = self.bucket.get_blob(blob_path)
            if not blob:
                logger.error("No blob found at path: %s", blob_path)
                return None

            data = blob.download_as_bytes()

            if json_type:
                return json.loads(data)

            return data

        except Exception as e:
            logger.error("An error occurred while retrieving blob data: %s", e)
            return None

    def get_data_mime_type(
        self,
        gsutil_uri: str,
    ) -> str:
        """
        Retrieves the MIME type of a blob from Google Cloud Storage.

        Parameters
        ----------
        gsutil_uri : str
            The Google Cloud Storage URI of the blob.

        Returns
        -------
        str
            The MIME type of the file in the blob.
        """
        if not gsutil_uri.startswith(f"gs://{self.bucket_name}/"):
            logger.error(
                "Invalid GSURI format. It should start with 'gs://your-bucket-name/'"
            )
            return None

        # Extract blob path from the GS URI
        blob_path = gsutil_uri.removeprefix(f"gs://{self.bucket_name}/")

        try:
            blob = self.bucket.blob(blob_path)
            if not blob.exists():
                logger.error("Blob not found at path: %s", blob_path)
                return None
            # If content_type is None, try to guess from the filename
            if blob.content_type is None:
                mime_type, _ = mimetypes.guess_type(blob.name)
                if mime_type:
                    logger.warning(
                        "MIME type not set for %s, guessing: %s", gsutil_uri, mime_type
                    )
                    return mime_type
                logger.warning("Unable to determine MIME type for %s", gsutil_uri)
                return None

            return blob.content_type

        except Exception as e:
            logger.error("An error occurred while retrieving blob MIME type: %s", e)
            return None



generative_model
"""
A class to convert multimedia data into text by usin generative models
"""

from typing import Literal, List, Union
import logging
import json

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part

from wxprocessing.data.make_dataset import GetCloudStorageData
from wxprocessing.credentials import vertex_credentials

# Get module logger
logger = logging.getLogger(__name__)

class VertexAnalysis(GetCloudStorageData):
    """
    A class to describe media content and perform sentiment analysis on chat conversations
    using Vertex AI generative models.

    Attributes
    ----------
    gen_model_name : str
        The name of the Vertex AI Generative Model to use.
    model : GenerativeModel
        The Vertex AI Generative Model used for generating multimedia descriptions.
    text_instruction : str
        The instruction to be used by the generative model to perform sentiment analysis.
    generation_config : dict
        Configuration for the generative model's output.
    safety_settings : dict
        Safety settings for the generative model to handle harmful content.
    audios_prompt : str
        Prompt for generating audio summaries.
    images_prompt : str
        Prompt for generating image descriptions.
    videos_prompt : str
        Prompt for generating video summaries.
    documents_prompt : str
        Prompt for generating document summaries.
    """

    def __init__(
        self,
        gen_model_name: str = "gemini-1.5-flash-001",
        sentiment_analysis: bool = True,
    ):

        super().__init__()

        # Initialize Vertex AI
        vertexai.init(
            project="dolphin-prod",
            location="us-central1",
            credentials=vertex_credentials,
        )

        # Set the model name
        self.gen_model_name = gen_model_name

        self.text_instruction = """
        Task: Act as an expert in Sentiment Analysis.
        
        Instructions: Extract the following information from the chat conversation:
        1. The customer's sentiment ("positivo," "negativo," or "neutral") at the beginning of the conversation.
        2. A brief explanation justifying the assigned sentiment at the beginning of the conversation.
        3. The customer's sentiment ("positivo," "negativo," or "neutral") at the end of the conversation.
        4. A brief explanation justifying the assigned sentiment at the end of the conversation.
        
        Conversation Message Structure: Each message in the conversation has the following format:
        - <message timestamp> - from: <sender type> <sender name> - message: <message content>.
        - If the message contains multimedia data, it can follow any of these formats:
            - <message timestamp> - from: <sender type> <sender name> - The following multimedia content is sent:., followed by the multimedia data. For messages where the data was extracted successfully.
            - <message timestamp> - from: <sender type> <sender name> - The multimedia content could not be extracted from the message. For messages where the multimedia data could not be extracted since it could have been corrupt or empty.
                
        Output Format: Provide the output in JSON format, using keys in English and values in Spanish, as shown in the examples below:
        
        Example 1:
        ```json
        {
            "initial_sentiment": "neutral",
            "initial_sentiment_justification": "El cliente busca ayuda para acceder a los servicios de la plataforma PTM ya que perdió sus credenciales.",
            "final_sentiment": "positivo",
            "final_sentiment_justification": "El cliente recibió la asesoría necesaria para reestablecer sus credenciales y acceder a los servicios de la plataforma PTM."
        }
        ```
        
        Example 2:
        ```json
        {
            "initial_sentiment": "negativo",
            "initial_sentiment_justification": "El cliente expresa inconformidad ya que ha realizado una recarga que no se ha reflejado en el saldo de su cuenta en la plataforma PTM.",
            "final_sentiment": "negativo",
            "final_sentiment_justification": "No se pudo solucionar el problema con la recarga ya que no se encontró comprobante de esta."
        }
        ```
        
        Example 3:
        ```json
        {
            "initial_sentiment": "neutral",
            "initial_sentiment_justification": "El cliente quiere que le activen uno de los productos de apuestas (BetPlay) que ofrece la plataforma PTM.",
            "final_sentiment": "negativo",
            "final_sentiment_justification": "No se le puede activar el producto al cliente ya que se encuentra en una región restringida, por lo que el cliente manifiesta su inconformidad ya que la competencia sí cuenta con la activación de este producto."
        }
        ```
        
        - initial_sentiment: Indicate the customer's sentiment (in Spanish) as "positivo," "negativo," or "neutral" at the beginning of the conversation.
        - initial_sentiment_justification: Provide a brief explanation in Spanish justifying the assigned sentiment at the beginning of the conversation.
        - final_sentiment: Indicate the customer's sentiment (in Spanish) as "positivo," "negativo," or "neutral" at the end of the conversation.
        - final_sentiment_justification: Provide a brief explanation in Spanish justifying the assigned sentiment at the end of the conversation.
        
        Ensure the explanations are concise and directly reflect the customer's expressed emotions or issues.
        """
        if sentiment_analysis:
            # Instantiate the generative model
            self.model = GenerativeModel(
                self.gen_model_name,
                system_instruction=[self.text_instruction],
            )
        else:
            # Instantiate the generative model
            self.model = GenerativeModel(
                self.gen_model_name,
            )

        # Generation configuration
        self.generation_config = {
            "max_output_tokens": 1024,
            "temperature": 1,
            "top_p": 0.95,
        }

        # Safety settings for the generative model
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        self.audios_prompt = """
        You are an expert at making audio transcription. Please generate a 
        transcription in Spanish of the following audio.
        """

        self.images_prompt = """
        You are an expert at describing images with certainty and detail. 
        Please generate a description of the following image in Spanish.
        """

        self.videos_prompt = """
        You are an expert at describing videos with certainty and detail. 
        Please generate a short summary in Spanish of the following video 
        including the most important ideas. If there are questions or requests 
        in the video please include them in the summary.
        """

        self.documents_prompt = """
        You are an expert at creating document summaries. Please generate a 
        brief summary in Spanish of the following document accurately describing 
        its content.
        """

    def format_conversation(
        self, conversation_list: List[dict]
    ) -> List[Union[str, Part]]:
        """
        Formats the conversation data into a list with the messages and multimedia data.

        Parameters
        ----------
        conversation_list : List[dict]
            A list of dictionaries containing the conversation data. Each dictionary should
            have the keys:
            - 'date': str, The timestamp of the message.
            - 'from': str, The type of sender (e.g., 'agent', 'customer').
            - 'from_name': str, The name of the sender.
            - 'message': str, The content of the message.

        Returns
        -------
        List[Union[str, Part]]
            A list containing strings (for text messages) and Part objects
            (for multimedia messages).
        """
        conversation_text = []
        for msg in conversation_list:
            msg_date = msg["date"]
            msg_from = msg["from"]
            msg_from_name = msg["from_name"]
            msg_content = msg["message"]

            if msg_content.startswith("gs://") and not msg_content.endswith(".json"):
                mime_type = self.get_data_mime_type(msg_content)
                if mime_type is None:
                    logger.warning("Could not determine MIME type for: %s", msg_content)
                    conversation_text.append(
                        f"{msg_date} - from: {msg_from} {msg_from_name} - message: {msg_content} [MIME type not found]"
                    )
                else:
                    conversation_text.append(
                        f"{msg_date} - from: {msg_from} {msg_from_name} - The following multimedia content is sent:"
                    )
                    conversation_text.append(
                        Part.from_uri(mime_type=mime_type, uri=msg_content)
                    )
            elif msg_content.startswith("gs://") and msg_content.endswith(".json"):
                conversation_text.append(
                    f"{msg_date} - from: {msg_from} {msg_from_name} - The multimedia content could not be extracted from the message"
                )
            else:
                conversation_text.append(
                    f"{msg_date} - from: {msg_from} {msg_from_name} - message: {msg_content}"
                )

        return conversation_text

    def generate_text_from_media(
        self,
        content: bytes,
        mime_type: str,
        media_type: Literal["audio", "image", "video", "document"],
    ) -> str:
        """
        Generates a description of multimedia content using Vertex AI Generative Models.

        Parameters
        ----------
        content : bytes
            The content of the multimedia to describe.
        mime_type : str
            The MIME type of the multimedia content.
        media_type : Literal["audio", "image", "video", "document"]
            The type of multimedia content (audio, image, video, document).

        Returns
        -------
        str
            A generated description of the multimedia content.

        Raises
        ------
        ValueError
            If an unsupported media type is provided.
        """
        # Build media part
        media = Part.from_data(
            data=content,
            mime_type=mime_type,
        )

        # Generate prompt according to the media type
        prompt = None
        if media_type == "audio":
            prompt = [self.audios_prompt, media]
        elif media_type == "image":
            prompt = [self.images_prompt, media]
        elif media_type == "video":
            prompt = [self.videos_prompt, media]
        elif media_type == "document":
            prompt = [self.documents_prompt, media]

        if prompt is None:
            raise ValueError(f"Specified media type '{media_type}' unsupported.")

        # Generate response
        responses = self.model.generate_content(
            contents=prompt,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            stream=True,
        )

        # Consuming the iterator
        text = ""
        for response in responses:
            # print(response.text, end="")
            text += response.text

        return text

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given conversation text using a Vertex AI generative model.

        The model is configured to classify the sentiment as positive, negative, or neutral,
        and provide a brief explanation for the classification.

        Parameters
        ----------
        text : str
            The conversation text to analyze.

        Returns
        -------
        dict
            A dictionary containing the sentiment classification and explanation.
        """

        # Generate response
        responses = self.model.generate_content(
            contents=text,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            stream=True,
        )

        # Format response
        sentiment = ""
        for response in responses:
            sentiment += response.text

        sentiment = sentiment.replace("```json", "").replace("```", "").strip()
        sentiment = json.loads(sentiment)

        return sentiment



