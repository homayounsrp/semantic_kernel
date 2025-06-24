# main.py
import asyncio
from kernel import create_kernel_with_history

from Config.llm import llm

async def main():
    # Build kernel, chat history, and streaming-enabled settings
    kernel, chat_history, execution_settings = create_kernel_with_history()

    # Instantiate our streaming assembler parser
    parser = StreamingAssemblerParser()

    while True:
        user_input = input("Enter your message >>> ")
        if user_input.lower() == "q":
            print("Exiting program.")
            break

        # Add user message to history
        chat_history.add_user_message(user_input)

        # Kick off streaming response
        stream = llm.get_streaming_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Accumulate the raw response so we can store it in history
        full_response = ""

        # As tokens arrive, pass each chunk to our parser and print immediately
        async for chunk in stream:
            if chunk.content:
                # parse_token_async returns the raw fragment for display
                token_to_display = await parser.parse_token_async(chunk)
                print(token_to_display, end="", flush=True)
                full_response += chunk.content

        # Once the stream completes, finalize parser state
        await parser.complete_processing_async()

        # Newline for readability after streaming ends
        print()

        # Add the assembled raw response to chat history
        chat_history.add_assistant_message(full_response)

if __name__ == "__main__":
    asyncio.run(main())
