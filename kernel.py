# kernel.py
from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

from Plugins import NapoleonPlugin
from Config.llm import llm

def create_kernel_with_history():
    """
    Build and return:
      - kernel: a Semantic Kernel instance with your Napoleon plugin
      - chat_history: an empty ChatHistory seeded with a system message
      - execution_settings: streaming-enabled chat settings
    """
    kernel = Kernel()

    # Register Napoleon plugin
    kernel.add_plugin(NapoleonPlugin(), plugin_name="get_vector_response")

    # Register the OpenAI LLM service
    kernel.add_service(llm)

    # Initialize chat history with a system prompt
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a historian agent. You can search in a vector database and retrieve historical info about Napoleon."
    )

    # Create streaming-enabled settings (no response_parser field here)
    execution_settings = OpenAIChatPromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
        stream=True
    )

    return kernel, chat_history, execution_settings
