from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from src.vector_db.query_embeddings import query_embeddings



def init_model(model="llama-3.3-70b-versatile"):
    return ChatGroq(model=model, temperature=0.0)


# Topic QA node
def topic_qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles user questions about a general topic (not a specific paper).
    Retrieves relevant chunks from the vector DB and synthesizes a response.
    Updates `last_bot_response` in the state.
    """
    # print("it's topic_qa node")
    llm = init_model()

    # Retrieve top 5 most similar chunks (adjust k as needed)
    query=state["chat_history"][-1]["content"]
    history=state["chat_history"][-3:-1]
    results = query_embeddings(query,n_results=5)

    retrieved_chunks: List[str] = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunk_text = f"{doc} (Title: {meta.get('title')}, Domain: {meta.get('domain')})"
        retrieved_chunks.append(chunk_text)

    if not retrieved_chunks:
        state["last_bot_response"] = "I couldn't find relevant information on that topic."
        return state

    # Build prompt for LLM
    chunks_text = "\n\n".join(retrieved_chunks[:5])  # top 5 chunks
    prompt_template = ChatPromptTemplate.from_template("""
You are an expert research assistant.

chat history:
{history}

The user asked:
"{user_query}"

You have access to the following relevant documents (from at least 1 paper) :

{retrieved_chunks}

Please provide a concise, informative, and coherent answer based only on the information above.
If the information is not available, say you don't know.
""")

    prompt_value = prompt_template.invoke({
        "user_query": query,
        "retrieved_chunks": chunks_text,
        "history":history
    })

    # Call the LLM
    try:
        response = llm.invoke(prompt_value)
        answer = response.content
    except Exception:
        answer = "Sorry, I couldn't generate an answer."

    # Update state
    state["chat_history"].append({"role":"assistant","content":answer.strip()})

    return state
