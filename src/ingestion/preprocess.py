import os
import json
from typing import List
from PyPDF2 import PdfReader
import nltk
import fitz
from tqdm import tqdm
import re 

from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("📥 Downloading required NLTK data...")
    nltk.download('punkt_tab', quiet=True)


def load_metadata(path="data/processed/metadata.json"):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)





def extract_abstract(pdf_path: str, max_chars_if_not_found: int = 2000) -> str:
    """
    Extract the abstract (or as close as possible) from a scientific PDF.

    - Tries to capture text after 'Abstract' until the next common heading.
    - If no obvious heading is found, returns up to max_chars_if_not_found chars.
    """
    # -------- 1) Read entire text -----------
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    doc.close()

    # -------- 2) Common section headings that usually follow the abstract ------
    # Add or remove depending on the papers you have
    next_headings = [
        "introduction", "background", "related work", "preliminaries",
        "methods", "materials", "approach", "problem formulation",
        "experimental", "experiments", "evaluation", "results",
        "discussion", "1 ", "I. ", "I ", "1.", "II ", "II.",
    ]
    # Build a single regex group with all these possibilities
    stop_pattern = "|".join(re.escape(h) for h in next_headings)

    # -------- 3) Regex to capture Abstract section ----------------------------
    # (?is) = case-insensitive + dot matches newlines
    pattern = re.compile(rf"(?is)abstract[:\s]*(.*?)(?=\n\s*(?:{stop_pattern}))")
    m = pattern.search(full_text)
    if m:
        abstract = m.group(1).strip()
    else:
        # Fallback: just grab some text after the first occurrence of 'abstract'
        m2 = re.search(r"(?is)abstract[:\s]*(.*)", full_text)
        if m2:
            abstract = m2.group(1).strip()
            # truncate if too long
            abstract = abstract[:max_chars_if_not_found]
        else:
            abstract = ""

    # -------- 4) Clean up spacing --------------------------------------------
    abstract = re.sub(r"\s+", " ", abstract)
    return abstract














def extract_text_from_pdf(pdf_path):

    try:
        reader=PdfReader(pdf_path)

        text=[]

        for page in reader.pages:
            content=page.extract_text()
            if content:
                text.append(content)
            
        return "\n".join(text)
    except Exception as e:
        return ""


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)   # collapse multiple spaces/newlines
    text = text.encode("utf-8", "ignore").decode()  # remove bad unicode
    return text.strip()


def chunk_text_with_overlap(text: str, max_tokens: int = 350, overlap: int = 50):
    sentences = sent_tokenize(text)
    chunks, current_chunk, token_count = [], [], 0

    for sent in sentences:
        words = sent.split()
        if token_count + len(words) > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            token_count = len(current_chunk)
        current_chunk.extend(words)
        token_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks







def process_pdfs(metadata_path,data_path,out_path):

    pdfs_metadata=load_metadata(metadata_path)


    all_chunks = []
    x=0
    for metadata in tqdm(pdfs_metadata):
        x+=1
        
        pdf_path=os.path.join(data_path,metadata["source_id"]+".pdf")

        if not os.path.exists(pdf_path):
            continue

        abstract_text = extract_abstract(pdf_path, max_chars_if_not_found=2000)
        
        raw_text=extract_text_from_pdf(pdf_path)

        cleaned_text=clean_text(raw_text)
        

        chunked_text=chunk_text_with_overlap(cleaned_text,350,50)

        for i, chunk in enumerate(chunked_text):


            if i==0:
                entry_id = f"{metadata['source_id']}_chunk_0"

                entry = {
                "id":entry_id,
                "source_id": metadata["source_id"],
                "domain": metadata["domain"],
                "citation_count": metadata["citation_count"],
                "date_published": metadata["date_published"],
                "chunk_index": 0,
                "text_chunk": metadata["title"]+ abstract_text,
                "metadata": {
                    "title": metadata["title"],
                    "authors": metadata["authors"],
                    "doi": metadata["doi"],
                },
                "is_abstract": True
                }
                all_chunks.append(entry)
            entry_id = f"{metadata['source_id']}_chunk_{i+1}"

            entry = {
                "id":entry_id,
                "source_id": metadata["source_id"],
                "domain": metadata["domain"],
                "citation_count": metadata["citation_count"],
                "date_published": metadata["date_published"],
                "chunk_index": i+1,
                "text_chunk": chunk,
                "metadata": {
                    "title": metadata["title"],
                    "authors": metadata["authors"],
                    "doi": metadata["doi"],
                },
                "is_abstract": False
            }
            all_chunks.append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in all_chunks:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    

        




def preprocess_papers(kind=[True,True]):
    
    if kind[0]:
        print("preprocessing important papers")
        process_pdfs("data/processed/metadata.json","data/raw","data/processed/chunks.jsonl")
    if kind[1]:
        print("preprocessing recent papers")
        process_pdfs("data/processed/metadata_recent.json","data/raw_recent","data/processed/chunks_recent.jsonl")



if __name__=="__main__":
    preprocess_papers()

