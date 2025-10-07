from langgraph.graph import StateGraph, START, END
from src.langgraph_workflow.intent_classifier import intent_classifier_node
from src.langgraph_workflow.paper_summarizer import abstract_formatter_node
from src.langgraph_workflow.period_trend_retriever import period_trend_retriever_node
from typing import TypedDict, Optional, List, Dict, Any
from src.langgraph_workflow.intent_classifier import IntentInfo
from src.langgraph_workflow.paper_details import paper_details_node,paper_determining_node
from src.langgraph_workflow.fallback_responder import fallback_responder_node
from src.langgraph_workflow.topic_qa import topic_qa_node

class State(TypedDict):
    query: Optional[str]
    intent_info: Optional[IntentInfo]
    period_papers:  List[Dict[str, Any]]
    chat_history: List[Dict[str, str]]   # [{"role":"user","content":...}, ...]
    last_bot_response: Optional[str]
    papers_retrieved: List[Dict[str, Any]]   



def create_graph():
    graph = StateGraph(State)          # StateGraph(State) is used to create the graph

    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("abstract_formatter", abstract_formatter_node)
    graph.add_node("paper_determiner", paper_determining_node)
    graph.add_node("fallback_responder", fallback_responder_node)
    graph.add_node("topic_retriever",topic_qa_node)
    graph.add_node("paper_details",paper_details_node)

    graph.add_node("period_trend_retriever", period_trend_retriever_node)

    graph.add_edge(START, "intent_classifier")

    graph.add_edge("fallback_responder", END)
    graph.add_edge("period_trend_retriever", "abstract_formatter")
    graph.add_edge("abstract_formatter", END)

    graph.add_conditional_edges(
        "intent_classifier",
        lambda state: state["intent_info"].intent,  # Dictionary access
        {
            "latest_papers": "period_trend_retriever",
            "topic_qa": "topic_retriever",
            "specific_paper": "paper_determiner",
            "out_of_scope": "fallback_responder",
        },
    )
    graph.add_edge("paper_determiner","paper_details")
    graph.add_edge("paper_details",END)
    graph.add_edge("fallback_responder",END)
    graph.add_edge("topic_retriever",END)
    compiled_graph = graph.compile()
    return compiled_graph