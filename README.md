﻿# Semantic Kernel Napoleon Historian Agent

This project is an AI-powered historian agent focused on Napoleon, leveraging Semantic Kernel, OpenAI, and Neo4j. It enables users to ask historical questions, which are answered using a vector database and language models.

## Features

- **Conversational agent**: Chat with an AI historian about Napoleon.
- **Vector search**: Retrieves relevant historical information from a Neo4j-backed vector database.
- **Streaming responses**: See answers as they are generated in real time.


## Setup

1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
3. **Configure environment variables**:  
   Copy `.env` and fill in your credentials for Neo4j and OpenAI.

4. **Run the agent**:
    ```sh
    python main.py
    ```

## Environment Variables

Edit `.env` with your credentials:
NEO4J_URI= 
NEO4J_USERNAME= 
NEO4J_PASSWORD= 
NEO4J_DATABASE= 
OPENAI_API_KEY=



## Usage

- Start the program and enter your historical question about Napoleon.
- Type `q` to exit.

## License

MIT License
