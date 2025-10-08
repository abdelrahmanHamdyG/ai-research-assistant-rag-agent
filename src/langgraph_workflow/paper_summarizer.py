# src/langgraph_workflow/abstract_formatter_node.py

from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# ---------- LLM Init ----------
def init_model(model: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
    """
    Initialize the Groq LLM client.
    """
    return ChatGroq(model=model, temperature=temperature)

# ---------- Prompt Template ----------
summary_prompt = ChatPromptTemplate.from_template("""
You are an expert research assistant.

This is the chat history:
{chat_history}

You retrieved the following papers with abstracts:
{papers_list}

For **each** paper (modify based on the user query):
- Summarize the abstract in 1–2 concise sentences.
- Keep it informative and professional.
- Output a numbered list like this:

1) Paper Title: <title>
   Summary: <summary>
2) Paper Title: <title>
   Summary: <summary>
...

User query: {query}
Respond in a friendly way to the user.
""")
# ---------- Node ----------
def abstract_formatter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node:
      • Takes state['user_query'] and state['papers_retrieved'] (list of papers).
      • Generates a user-friendly output with summaries of each paper.
      • Stores the output in state['final_output'].
    """
    llm = init_model()
    
    papers_retrieved: List[Dict[str, Any]] = state.get("papers_retrieved", [])
    
    # Build a plain text list of titles + abstracts for LLM
    papers_text = "\n\n".join(
        [f"{i+1}) Title: {p['title']}\nAbstract: {p['abstract']}" for i, p in enumerate(papers_retrieved[:10])]
    )

    # Format prompt
    history=state.get("chat_history", [])[-3:]
    query=state.get("chat_history", [])[-1]["content"]
    prompt_value = summary_prompt.format_prompt(
        chat_history=history,
        papers_list=papers_text,
        query=query
    )


    # LLM call
    resp = llm.invoke(prompt_value)

    # Store the final formatted text in state
    state["chat_history"].append({"role":"assistant","content":resp.content.strip()})
    return state
