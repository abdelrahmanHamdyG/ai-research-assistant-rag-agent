from ast import List
import requests
import yaml
from requests.adapters import HTTPAdapter
import os 
import re 
from datetime import datetime, timedelta
import json
from tqdm import tqdm 
import time
import arxiv

with open("config.yaml","r") as f:
    config=yaml.safe_load(f)

all_domains=list(map(lambda x:x["id"],config["concepts"]))

def is_pdf_url(url: str, timeout: int = 8) -> bool:
    """
    Quick check if URL returns a PDF - streamlined version
    """
    if not url:
        return False
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Try HEAD first (fastest)
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        if r.status_code == 200:
            ctype = r.headers.get("Content-Type", "").lower()
            if "pdf" in ctype:
                return True
        
        # If HEAD unclear, try partial GET
        headers['Range'] = 'bytes=0-100'
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code in [200, 206]:
            if r.content.startswith(b'%PDF'):
                return True
        
        return False
        
    except Exception:
        return False

def safe_name(name):
    # Handle None or non-string values
    if name is None:
        return "unnamed_paper"
    if not isinstance(name, str):
        name = str(name)
    
    # Original logic
    name = re.sub(r'[^\w\-_. ]', '_', name)
    return name.replace(" ", "_")


def find_best_pdf_url(work):
    """
    Streamlined PDF finder - only the most effective patterns
    """
    potential_urls = []
    
    # 1. Get all OpenAlex URLs
    if work.get('open_access', {}).get('oa_url'):
        potential_urls.append(work['open_access']['oa_url'])
    
    best_oa = work.get('best_oa_location', {})
    if best_oa and best_oa.get('url'):
        potential_urls.append(best_oa['url'])
    
    # Get ALL locations, prioritize OA
    locations = sorted(work.get('locations', []), 
                      key=lambda x: x.get('is_oa', False), reverse=True)
    
    for loc in locations:
        if loc.get('url'):
            potential_urls.append(loc['url'])
    
    # 2. Convert to PDF URLs with ONLY the most reliable patterns
    pdf_urls = []
    for url in potential_urls:
        if not url:
            continue
            
        # ArXiv - most reliable
        if 'arxiv.org/abs/' in url:
            pdf_urls.append(url.replace('arxiv.org/abs/', 'arxiv.org/pdf/') + '.pdf')
        
        # PMC - very reliable
        elif 'ncbi.nlm.nih.gov/pmc/articles/' in url and not url.endswith('.pdf'):
            pmc_id = url.split('/')[-1].rstrip('/')
            if pmc_id.startswith('PMC'):
                pdf_urls.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/")
        
        # MDPI - very common for recent papers
        elif 'mdpi.com' in url and not url.endswith('.pdf'):
            pdf_urls.append(url + '/pdf')
        
        # CVPR/CVF - reliable for CV papers
        elif 'openaccess.thecvf.com' in url:
            if url.endswith('.html'):
                pdf_urls.append(url.replace('.html', '.pdf'))
            elif not url.endswith('.pdf'):
                pdf_urls.append(url + '.pdf')
        
        # OpenReview - good for recent ML
        elif 'openreview.net' in url and '/forum?id=' in url:
            paper_id = url.split('id=')[-1].split('&')[0]
            pdf_urls.append(f"https://openreview.net/pdf?id={paper_id}")
        
        # PMLR - ML proceedings
        elif 'proceedings.mlr.press' in url and not url.endswith('.pdf'):
            pdf_urls.append(url + '.pdf')
        
        # Always try original URL
        pdf_urls.append(url)
    
    # 3. Test URLs quickly
    for url in pdf_urls[:10]:  # Limit to first 10 to save time
        if is_pdf_url(url):
            return url
    
    return None

def get_arxiv_link(work: dict):
    """
    Simplified arXiv search - just the most effective approach
    """
    title = work.get("title") or work.get("display_name")
    if not title:
        return None

    try:
        # Only try exact title search
        search = arxiv.Search(
            query=f'ti:"{title}"',
            max_results=3,  # Reduced for speed
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in search.results():
            # Simple word overlap check
            result_words = set(result.title.lower().split())
            query_words = set(title.lower().split())
            overlap = len(result_words & query_words) / len(query_words | result_words)
            
            if overlap > 0.7:  # High threshold for accuracy
                return result.pdf_url

    except Exception:
        pass

    return None

def get_semantic_scholar_pdf(work):
    """
    Simplified Semantic Scholar - one quick attempt
    """
    title = work.get("title") or work.get("display_name")
    if not title:
        return None
    
    try:
        params = {
            'query': title[:100],  # Truncate long titles
            'limit': 3,  # Reduced for speed
            'fields': 'openAccessPdf,title'
        }
        
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params, 
            timeout=8
        )
        
        if response.status_code == 200:
            for paper in response.json().get('data', []):
                paper_title = paper.get('title', '')
                if paper_title:
                    # Quick similarity check
                    title_words = set(title.lower().split())
                    paper_words = set(paper_title.lower().split())
                    overlap = len(title_words & paper_words) / len(title_words | paper_words)
                    
                    if overlap > 0.6 and paper.get('openAccessPdf', {}).get('url'):
                        pdf_url = paper['openAccessPdf']['url']
                        if is_pdf_url(pdf_url):
                            return pdf_url
    
    except Exception:
        pass
    
    return None

def get_remote_file_size(url, timeout=5):
    """
    Try to fetch the file size (Content-Length) via HEAD request.
    Returns size in bytes or None if not available.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        if "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    return None

def download_pdf(pdf_url: str, name: str, out_dir="data/raw", 
                 max_size_mb=15, max_time=7, timeout=5):
    """
    Downloads a PDF safely
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, name + ".pdf")

    # 1. Check remote file size
    size = get_remote_file_size(pdf_url, timeout=timeout)
    if size and size > max_size_mb * 1024 * 1024:
        return "too_large"

    # 2. Download with streaming
    try:
        start = time.time()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        with requests.get(pdf_url, stream=True, timeout=timeout, allow_redirects=True, headers=headers) as r:
            r.raise_for_status()
            size_downloaded = 0
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        break
                    f.write(chunk)
                    size_downloaded += len(chunk)

                    if size_downloaded > max_size_mb * 1024 * 1024:
                        return "too_large"

                    if time.time() - start > max_time:
                        return "timeout"

        
        return "saved"

    except Exception as e:
        print(f"⚠️ Failed to download {pdf_url}: {e}")
        return "failed"

papers_ids=set()

def is_primary_concept(work,concept):
    needed_url=f"https://openalex.org/{str(concept)}"
    concepts=work["concepts"]
    for cur_concept in concepts:
        if cur_concept["id"]==needed_url:
            return True if cur_concept["score"]>0.45 else False
    return False

def fetch_paper_by_concept(id,domain,minimum_citations=400,max_results=1,recent_days=None,out_dir="data/raw"):
    global papers_ids
    base_url=config["sources"]["openalex"]["api_url"]


    if minimum_citations==0:
        params = {
            "filter": f"concepts.id:{id}",
            "sort": "publication_date:desc",
            "per-page": max_results,
        }


    else:    
        params = {
            "filter": f"concepts.id:{id},cited_by_count:>{minimum_citations}",
            "sort": "publication_date:desc",
            "per-page": max_results,
        }

    if recent_days:
        date_limit = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        params["filter"] += f",publication_date:>{date_limit}"

    papers=[]
    response=requests.get(base_url,params=params).json()
    response_results=response["results"]
    # print(f"Found {len(response_results)} papers for {domain}")

    for paper in response.get("results", []):
        if paper["id"] in papers_ids:
            continue
             
        source_id=paper["id"].split("/")[-1]
        metadata = {
            "source_id": source_id,
            "title": paper["display_name"],
            "doi": paper.get("doi"),
            "date_published": paper.get("publication_date"),
            "citation_count": paper.get("cited_by_count"),
            "authors": [a["author"]["display_name"] for a in paper.get("authorships", [])],
            "domain": domain,
            "name":safe_name(paper["display_name"])
        }
        
        

        if is_primary_concept(paper,id):
            
            
            pdf_url = None
            
            # Try 3 methods only - fast and effective
            
            pdf_url = find_best_pdf_url(paper)
            if not pdf_url:
                pdf_url = get_arxiv_link(paper)
                if not pdf_url:
                    pdf_url = get_semantic_scholar_pdf(paper)
                    
            
            if pdf_url:
                download_result = download_pdf(pdf_url, metadata["source_id"],out_dir=out_dir)
                if download_result == "saved":
                    papers_ids.add(metadata["source_id"])
                    papers.append(metadata)
                
    print(f"\n📊 {domain}: {len(papers)} papers collected")
    return papers

def fetch_recent_papers(days_back=7,number_of_citations=0):
    fetch_all_papers(minimum_citations=number_of_citations,max_results=50,recent_days=days_back,file_path="metadata_recent.json",out_dir="data/raw_recent")

def fetch_all_papers(minimum_citations=5000,max_results=30,file_path="metadata.json",recent_days=None,out_dir="data/raw"):
    all_papers=[]
    concepts=config["concepts"]

    for item in tqdm(concepts):
        id=item["id"]
        domain=item["domain"]
        domain_papers=fetch_paper_by_concept(id=id,domain=domain,minimum_citations=minimum_citations,max_results=max_results,recent_days=recent_days,out_dir=out_dir)
        all_papers.extend(domain_papers)

    os.makedirs("data/processed", exist_ok=True)
    with open(f"data/processed/{file_path}", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    

def fetch_papers(kind=[True,True]):
    if kind[0]:
        print("fetching important papers")
        fetch_all_papers()
    if kind[1]:
        print("fetching recent papers")
        fetch_recent_papers()

if __name__=="__main__":
    fetch_papers()