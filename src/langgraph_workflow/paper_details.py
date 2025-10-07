from typing import Dict, Any, List
from src.vector_db.query_embeddings import query_embeddings
from src.vector_db.query_embeddings import get_chunks_by_source_id_and_query
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import re
from langchain.output_parsers import PydanticOutputParser

# ---------- LLM Init ----------


class PaperMatch(BaseModel):
    paper_title: str = Field(..., description="Exact title of the matched paper, or 'null' if none")
    paper_id: str = Field(..., description="source_id of the matched paper, or 'null' if none")

def init_model(model: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
    """
    Initialize the Groq LLM client.
    """
    return ChatGroq(model=model, temperature=temperature)


def paper_determining_node(state: Dict[str, Any]) -> Dict[str, Any]:
  
    print("*********************************************")
    print("we are determining the paper ")
    print("*********************************************")
    if state["intent_info"].specific_paper_id is not None :
        print("we already know the paper ")
        return state
    else:
        paper_initial_title=state["intent_info"].specific_paper
        response=query_embeddings(query_text=paper_initial_title,n_results=4,is_abstract=[True])
        # state["intent_info"].specific_paper_id=response["metadatas"][0]["source_id"]
        # state["intent_info"].specific_paper=response["metadatas"][0]["title"]
        candidate_texts = response["documents"][0]
        candidate_metas = response["metadatas"][0]
        
        

        candidates = []
        print("candidate_metas is ",candidate_metas)
        for meta, abstr in zip(candidate_metas, candidate_texts):
            candidates.append({
                "title": meta.get("title"),
                "abstract": abstr,
                "source_id": meta.get("source_id")
            })
        match = match_specific_paper_with_llm(paper_initial_title, candidates)
        if match.paper_id.lower() != "null":
            state["intent_info"].specific_paper_id = match.paper_id
            state["intent_info"].specific_paper  = match.paper_title
            print("*********************************************")
            print(f"the paper_title is {match.paper_title}")
            print("*********************************************")

        else:
            state["intent_info"].specific_paper_id = None
            state["intent_info"].specific_paper  = None

    
    
        
    return state



    
    

def paper_details_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use only the chunks of the identified paper to answer the user query.
    """
    print("*********************************************")
    print("We are getting the details of the node ")
    print("*********************************************")
    info = state["intent_info"]
    print("info is ",info)
    
    if not info.specific_paper:
        print("paper not identified yet\n")
        state["error"] = "Paper not identified yet."
        return state
    
    
    
    
    chunks = get_chunks_by_source_id_and_query(info.specific_paper_id,query_text=state["chat_history"][-1]["content"],n_results=4)
    
    paper_text = "\n\n".join(chunks["documents"][0])


    print("*********************************************")
    print(paper_text)
    print("*********************************************")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful research assistant.
    Answer the following question **using only** the provided paper text.
    if you don't know the answer please say I don't know the answer 

    Chat History:{chat_history}

    Paper: {paper_text}

    Question: {question}
    """)


    chat_history=state["chat_history"][-3:-1]
    user_query = state["chat_history"][-1]["content"]
    
    prompt_value = prompt.format_prompt(paper_text=paper_text, question=user_query,chat_history=chat_history)


    
    resp = llm.invoke(prompt_value)

    state["chat_history"].append({"role":"assistant","content":resp.content.strip()})

    
    return state










def init_model(model="llama-3.3-70b-versatile"):
    return ChatGroq(model=model, temperature=0.0)


def match_specific_paper_with_llm(
    user_description: str,
    candidate_papers: List[Dict[str, Any]],
    llm=None
) -> PaperMatch:
    """
    Ask the LLM to choose the candidate paper that best matches the user description.
    Each candidate dict must have: 'title', 'abstract', 'source_id'.
    """
    if llm is None:
        llm = init_model()

    # Build readable candidate list with short abstract snippets
    def first_words(text: str, n: int = 80):
        words = re.split(r"\s+", text or "")
        return " ".join(words[:n])

    candidates_text = "\n\n".join(
        [
            f"""{i+1}) Title: {c.get('title', 'N/A')}
   Abstract snippet: {first_words(c.get('abstract', ''))}
   Source ID: {c.get('source_id', 'N/A')}"""
            for i, c in enumerate(candidate_papers)
        ]
    )

    prompt_template = ChatPromptTemplate.from_template("""
You are an expert research assistant.

The user is asking about a paper described as:
"{user_description}"

Here are candidate papers:
{candidates}

Choose the single paper that best matches the user description.

Return **only** valid JSON in this format:

{{
  "paper_title": "<exact candidate title>",
  "paper_id": "<candidate source_id>"
}}

If no match, return:

{{
  "paper_title": "null",
  "paper_id": "null"
}}
""")

    prompt_value = prompt_template.invoke({
        "user_description": user_description,
        "candidates": candidates_text
    })

    print("***************************")
    print("prompt_value is", prompt_value)
    print("***************************")

    parser = PydanticOutputParser(pydantic_object=PaperMatch)

    try:
        response = llm.invoke(prompt_value)
        # print("LLM raw response:", response.content)

        # Try to extract JSON if LLM added extra text
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        clean_response = json_match.group(0) if json_match else response.content.strip()

        result = parser.parse(clean_response)
    except Exception as e:
        print("⚠️ Exception while parsing LLM output:", e)
        result = PaperMatch(paper_title="null", paper_id="null")

    return result
