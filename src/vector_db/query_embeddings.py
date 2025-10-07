# src/vector_db/query_embeddings.py
import os
from chromadb import PersistentClient
from typing import List


def init_chroma(db_path="embeddings", collection_name="paper"):
    """
    Connect to the persistent Chroma DB and return the collection.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"❌ Database path '{db_path}' not found. Did you run the embedding script first?"
        )

    client = PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    return collection



def get_papers_abstract(date:int,domains:List[str],db_path: str = "embeddings", collection_name: str = "paper"):

    collection=init_chroma(db_path,collection_name)
    print(date)
    print("******")
    results = collection.get(
        where={
                "$and": [
                    {"is_abstract": {"$eq": True}},
                    {"paper_domain": {"$in": domains}},            # filter by specific domain
                    {"date_publushed_int": {"$gte": date}}  

                ]
            }
    )
    
    
    print(f"Total matching chunks: {len(results['ids'])}")
    return results
    
        

def get_chunks_by_source_id_and_query(
    source_id: str,
    query_text: str,
    n_results: int = 5,
    db_path: str = "embeddings",
    collection_name: str = "paper"
) :
    """
    Retrieve chunks for a specific paper (source_id) but
    also rank/filter them semantically by the given query.
    """
    collection = init_chroma(db_path=db_path, collection_name=collection_name)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"source_id": source_id},   # restrict to the paper
    )
    return results



def query_embeddings(
    query_text: str,
    n_results: int = 5,
    db_path: str = "embeddings",
    collection_name: str = "paper",
    is_abstract=[True,False]
):
    """
    Query the Chroma vector DB for the most similar chunks.

    Returns:
        dict with 'documents', 'metadatas', 'ids', 'distances'
    """
    collection = init_chroma(db_path=db_path, collection_name=collection_name)

    print(f"🔍 Searching for: {query_text}")
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where={ "is_abstract": { "$in": is_abstract} }   # <-- filter here

    )
    print("result size is ",len(results["documents"]))
    return results


def pretty_print(results: dict):
    """
    Nicely display the query results.
    """
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        print("=" * 60)
        print(f"Result #{i}")
        print(f"Distance: {dist:.4f}")
        print(f"Title   : {meta.get('title')}")
        print(f"Authors : {', '.join(meta.get('authors', []))}")
        print(f"DOI     : {meta.get('doi')}")
        print(f"Source  : {meta.get('source_id')}")
        print("---- Chunk Text ----")
        print(doc[:500] + ("..." if len(doc) > 500 else ""))


def main():
    # 🔧 Example usage
    query = input("Enter your search query: ").strip()
    results = query_embeddings(query_text=query, n_results=5)
    pretty_print(results)


if __name__ == "__main__":
    main()
