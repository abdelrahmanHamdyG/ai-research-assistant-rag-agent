from datetime import datetime,timedelta
import json,os
from typing import Dict,Any,List
from src.vector_db.query_embeddings import get_papers_abstract



def get_target_day_date(days_back:str):
    cutoff_int = int((datetime.now() - timedelta(days=7)).strftime("%Y%m%d"))
    return cutoff_int


def period_trend_retriever_node(state: Dict[str, Any]):
    intent_info = state["intent_info"]

    domains = intent_info.domains
    period = intent_info.period or 7
    print(f"period is {period} domain is {domains}")
    print("****************")


    target_date = get_target_day_date(period)
    papers_retrieved = get_papers_abstract(date=target_date, domains=domains)

    abstracts = papers_retrieved["documents"]
    ids = papers_retrieved["ids"]
    metadatas = papers_retrieved["metadatas"]

    # ✅ Combine into a single list of dicts
    papers = []
    for abs_text, paper_id, meta in zip(abstracts, ids, metadatas):
        # print(meta.get("title"))
        papers.append({
            "id": paper_id,
            "title": meta.get("title"),
            "abstract": abs_text
        })
    
    state["papers_retrieved"] = papers
    
    return state







    





    





