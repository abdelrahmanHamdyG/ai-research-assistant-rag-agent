from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


def init_model(model="llama-3.3-70b-versatile"):
    return ChatGroq(model=model, temperature=0.0)

def fallback_responder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles out-of-scope or casual messages with a concise LLM-based reply.
    """

    llm=init_model()
        

    query = state.get("chat_history", [])
    if len(query):
        query=query[-1]["content"]

    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful AI research assistant.
    The user's question is out of your research scope.
    Reply naturally and politely and mention that you are ai research assistant if needed, be concise. 
    If you don't know the answer, simply say you don't know the answer and you are AI reseach assistant bot.

    User query: {query}
    """)

    prompt_value = prompt_template.invoke({"query": query})

    try:
        response = llm.invoke(prompt_value)
        bot_reply = response.content.strip()
    except Exception:
        bot_reply = "I'm not sure about that, but I can help you with research topics or recent papers."

    
    
    state["chat_history"].append({"role":"assistant","content":response.content.strip()})
    return state
