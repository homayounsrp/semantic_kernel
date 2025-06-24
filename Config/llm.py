from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings
)
import os
from dotenv import load_dotenv
load_dotenv()

llm = OpenAIChatCompletion(
    service_id="openai-chat",
    ai_model_id="gpt-4.1-nano",   
    api_key=os.getenv('OPENAI_API_KEY')
)