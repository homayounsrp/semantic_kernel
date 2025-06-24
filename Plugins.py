
from semantic_kernel.functions import kernel_function,KernelArguments
from Config.VectorRAG import query_vector_rag
from Config.VectorRAG import query_vector_rag

class NapoleonPlugin:
    @kernel_function(description="Search for information in database; returns the response to user.")
    def get_vector_response(self, arguments: KernelArguments, top_k: int = 5) -> str:
        """
        Use this to get a vector response from the database.
        The user’s raw question text is expected under the key 'request' in KernelArguments.
        """

        # 1. Extract the user's question from the KernelArguments dict.
        question = arguments.get("request", "")
        if not isinstance(question, str) or question.strip() == "":
            return "I need a question to search for—what would you like to know?"

        # 2. Call your existing VectorRAG function, which expects a plain str.
        docs_and_scores = query_vector_rag(
            question=question,
            vector_index_name="Chunk",
            vector_node_label="Chunk",
            vector_source_property="text",
            vector_embedding_property="textEmbeddingOpenAI",
            top_k=top_k
        )

        # 3. If nothing was found, return a friendly message.
        if not docs_and_scores:
            return "Sorry, I couldn’t find any matching content."

        # 4. Otherwise, format the top-k results into a single string to return to SK.
        response_parts = []
        for idx, (doc, score) in enumerate(docs_and_scores, start=1):
            snippet = doc.page_content.strip().replace("\n", " ")
            response_parts.append(f"{idx}. {snippet}  (score: {score:.2f})")

        return "\n\n".join(response_parts)
