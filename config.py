import os
from typing import Optional
from dotenv import load_dotenv
import logging
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
import PyPDF2
import io
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ConfigError(Exception):
    """Custom exception for configuration errors"""

    pass


def get_api_key() -> str:
    """
    Retrieve OpenAI API key from environment variables.

    Returns:
        str: The OpenAI API key

    Raises:
        ConfigError: If API key is not found
    """
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, TypeError):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ConfigError(
                "OPENAI_API_KEY not found in environment variables or Streamlit secrets"
            )
    return api_key


def create_llm() -> ChatOpenAI:
    """
    Create and configure a LangChain OpenAI chat model.

    Returns:
        ChatOpenAI: Configured LLM instance

    Raises:
        ConfigError: If API key cannot be loaded
    """
    try:
        api_key = get_api_key()
        os.environ["OPENAI_API_KEY"] = api_key

        # Get model configuration from environment variables or use defaults
        model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
        temperature = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
        max_tokens = int(os.getenv("MAX_TOKENS", "1000"))

        logger.info(f"OpenAI API key loaded successfully, using model: {model_name}")
        return ChatOpenAI(
            model=model_name, temperature=temperature, max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"Failed to create LLM: {str(e)}")
        raise ConfigError(f"Failed to initialize language model: {str(e)}")


# State definition for the workflow
class State(TypedDict):
    """Type definition for the workflow state"""

    applicant: str
    job_requirements: str
    experience_level: str
    skill_match: str
    response: str
    error: Optional[str]


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text content from a PDF file using PyPDF2.

    This function extracts text from uploaded PDF files for resume parsing.
    It handles potential issues with scanned or protected PDFs.

    Args:
        pdf_file: The uploaded PDF file object from Streamlit

    Returns:
        str: Extracted text from the PDF

    Raises:
        ConfigError: If PDF extraction fails or returns empty text
    """
    try:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))

        # Extract text from all pages
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Check if any text was extracted
        if not text.strip():
            logger.warning("No text extracted from PDF, may be scanned or protected")
            raise ConfigError(
                "Could not extract text from the PDF. The file may be scanned or protected."
            )

        return text
    except ImportError:
        logger.error("PyPDF2 not installed")
        raise ConfigError(
            "PyPDF2 is not installed. Please install it with 'pip install PyPDF2'"
        )
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise ConfigError(f"Failed to extract text from PDF: {str(e)}")
