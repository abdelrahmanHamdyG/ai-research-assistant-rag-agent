from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv


load_dotenv()
INTENTS = ["latest_papers", "topic_qa", "specific_paper", "out_of_scope"]
DOMAINS = ["CV", "DL", "MM", "NLP", "ML"]

class IntentInfo(BaseModel):
    intent: str
    domains: List[str]
    period: Optional[int] = None
    specific_paper: Optional[str] = None
    specific_paper_id:Optional[str]=None

def init_model(model="llama-3.3-70b-versatile"):
    return ChatGroq(model=model, temperature=0.0)

def intent_classifier_node(state: Dict[str, Any], llm=None) -> Dict[str, Any]:
    """
    LangGraph-ready node to classify intent and domains using LLM + Pydantic structured output.
    Updates the shared state.
    """

    # print(" we are classifying intent ")
    if llm is None:
        llm = init_model()

    history = state.get("chat_history", [])
    
    hist=history[-3:-1]
    query=history[-1]["content"]
    # print(query)
    if not query:
        print("yes it's not query")
        state["intent_info"] = IntentInfo(intent="out_of_scope", domains=[], period=None, specific_paper=None)
        return state
    
    prompt_template = ChatPromptTemplate.from_template("""
You are an expert research assistant.
Classify the user's query into intent and domain(s) based on allowed values.

Previous paper info (can be null):
- Previous paper title: {prev_paper_title}
- Previous paper ID: {prev_paper_id}

Instructions:
1. If the user is still asking about the same paper, reuse the previous paper title and ID.
2. If the user asks about a new paper, update specific_paper and make specific_paper_id NULL.
3. If the query is not about a paper, set specific_paper and specific_paper_id to null.

Respond with JSON following this schema:
- intent: one of {INTENTS}
- domains: list of allowed domains {DOMAINS}
- period: last number of days mentioned (can be None)
- specific_paper: paper title if mentioned or its description, else null
- specific_paper_id: paper id if mentioned, else null

Chat history :
{history}

Query:
{query}
""")

    
    prev_paper_title=state["intent_info"].specific_paper if state["intent_info"] is not None else None
    prev_paper_id=state["intent_info"].specific_paper_id if state["intent_info"] is not None else None
    prompt_value = prompt_template.invoke({"INTENTS":INTENTS,"DOMAINS":DOMAINS,"query": query,"prev_paper_title":prev_paper_title,"prev_paper_id":prev_paper_id,"history":hist})
    
    # Use PydanticOutputParser instead of with_structured_output
    parser = PydanticOutputParser(pydantic_object=IntentInfo)

    try:
        response = llm.invoke(prompt_value)
        
        intent_info = parser.parse(response.content)
    except Exception:
        # print("here is the exception")
        intent_info = IntentInfo(intent="out_of_scope", domains=[], period=None, specific_paper=None)

    # Safety check  s
    # print(intent_info.intent)
    if intent_info.intent not in INTENTS:
        # print(f"yes it's not in it's actually {intent_info.intent}")
        intent_info.intent = "out_of_scope"
    intent_info.domains = [d for d in intent_info.domains if d in DOMAINS]

    
    state["intent_info"] = intent_info
    # print("intent_info is ",intent_info)
    state["last_query"] = query
    return state
