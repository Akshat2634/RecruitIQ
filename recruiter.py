import os
from dotenv import load_dotenv
import logging
from langchain_openai.chat_models import ChatOpenAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

os.environ["OPENAI_API_KEY"] = api_key

logger.info("OPENAI_API_KEY loaded successfully")

llm = ChatOpenAI(model="gpt-4o-mini")

llm.invoke("Hello, world!")