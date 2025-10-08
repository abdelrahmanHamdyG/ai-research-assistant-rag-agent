import json
from sentence_transformers import SentenceTransformer
from itertools import islice
import os 
from chromadb import PersistentClient
from src.vector_db.domain_classifier import classify_paper
from datetime import date, datetime



def date_str_to_int(date_str: str) -> int:
    """
    Convert a date string 'YYYY-MM-DD' to an integer YYYYMMDD.

    Examples
    --------
    >>> date_str_to_int("2025-09-24")
    20250924
    """
    # Validate and convert
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.strftime("%Y%m%d"))




def load_chuncks(file_path1="data/processed/chunks.jsonl",file_path2="data/processed/chunks_recent.jsonl",kind=[True,True]):

    
    if kind[0]:
        #print(" storing chunks for important papers ")
        with open(file_path1,"r",encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    if kind[1]:
        #print(" storing chunks for recent papers ")
        with open(file_path2,"r",encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    
def batched(iterable,batch_size=50):

    it=iter(iterable)
    batch=[]

    while batch:=list(islice(it,batch_size)):
        yield batch



def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    #print(f"🔄 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def init_chroma(db_path="embeddings", collection_name="paper"):
        
    os.makedirs(db_path, exist_ok=True)
    client = PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection



def store_embedding(kind):

    embedding_model=load_embedding_model()


    collection  =init_chroma()
    source_to_domain={}

    for batch in batched(load_chuncks(kind=kind),batch_size=50):
        ids, texts,metadatas =[],[],[]


        paper_domain="not_abstract"
        for i,chunk in enumerate(batch):
            
            chunk_id=chunk["id"]
            source_id=chunk["source_id"]
            
            if source_id in source_to_domain:
                    paper_domain=source_to_domain[source_id]
            else:
                if chunk["chunk_index"]==0:
                    
                    paper_domain=classify_paper(chunk["text_chunk"])
                    source_to_domain[source_id]=paper_domain
                


            metadata = {
                "source_id": chunk["source_id"],
                "domain": chunk["domain"],
                "citation_count": chunk["citation_count"],
                "date_published": chunk["date_published"],
                "date_publushed_int":date_str_to_int(chunk["date_published"]),
                "title": chunk["metadata"]["title"],
                "authors": ", ".join(chunk["metadata"]["authors"]),
                "doi": chunk["metadata"]["doi"],
                "paper_domain":paper_domain,
                "is_abstract":chunk["is_abstract"]
            }

            ids.append(chunk_id)
            texts.append(chunk["text_chunk"])
            metadatas.append(metadata)


        embeddings=embedding_model.encode(texts,show_progress_bar=True,convert_to_numpy=True)

        collection.upsert(ids=ids,embeddings=embeddings.tolist(),documents=texts,metadatas=metadatas)
        #print(f"stored {len(ids)} chunks ")



            



def store_papers_embedding(kind=[True,True]):
    store_embedding(kind)


if __name__=="__main__":
    store_papers_embedding()
    

