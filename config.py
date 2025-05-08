import os
from typing import Optional
from dotenv import load_dotenv
import logging
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI

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
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ConfigError("OPENAI_API_KEY not found in environment variables")
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
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
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
