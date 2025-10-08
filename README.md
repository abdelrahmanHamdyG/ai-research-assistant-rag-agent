# 🧠 AI Research Assistant (RAG Agent)

An intelligent AI Research Assistant powered by LangGraph, LangChain, and Groq LLMs, designed to help you explore, summarize, and interact with academic papers. It combines retrieval-augmented generation (RAG) with a structured reasoning graph to handle queries like finding trending papers, summarizing specific research, or answering topic-based questions.

## ⚙️ How It Works

The system runs as a LangGraph workflow, where each node is responsible for a different reasoning step — from classifying intent to fetching papers and generating summaries.

## 📊 Workflow Graph

![Workflow Graph](https://github.com/abdelrahmanHamdyG/ai-research-assistant-rag-agent/blob/main/graph.png)

## 🔹 Nodes Overview

| Node | Description |
|------|-------------|
| **Intent Classifier** | Detects whether the query is about latest papers, a specific paper, or a topic. |
| **Period Trend Retriever** | Fetches the most recent and relevant papers. |
| **Abstract Formatter** | Summarizes retrieved papers concisely. |
| **Topic QA** | Answers domain-specific questions using RAG. |
| **Paper Identifier** | Locates and matches a specific paper by title or description. |
| **Paper Details** | Provides in-depth answers about the chosen paper. |
| **Fallback Handler** | Responds gracefully when a query is out of scope. |

## 🧩 Data & Updates

* The assistant uses chunked embeddings of key AI research papers — including high-impact works with over 5000 citations.
* It also fetches and updates new recent papers periodically, removing old ones and refreshing the embeddings to ensure up-to-date knowledge.

## 💬 Example Conversations

**User:**  
*What are the top 3 papers in the last week in Computer Vision?*

**Bot:**  
I've got the latest updates in Computer Vision for you. Here are the top 3 papers from the last week, summarized for your convenience:

1. Effective image compression using transformer and residual network...
2. PyramidStyler: Transformer-Based Neural Style Transfer...
3. LCW-YOLO: A Lightweight Multi-Scale Object Detection Method...

---

**User:**  
*Can you summarize the U-Net paper and tell me what batch size was used?*

**Bot:**  
The U-Net paper presents a convolutional architecture for biomedical image segmentation, featuring a contracting path for context and an expanding path for localization. It was trained with a batch size of 1, prioritizing large input tiles over large batches to better utilize GPU memory.

## 🚀 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/abdelrahmanHamdyG/ai-research-assistant-rag-agent.git
cd ai-research-assistant-rag-agent
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4️⃣ Run the assistant

```bash
python -m src.main
```

## 🧠 Tech Stack

* **LangGraph** – structured workflow orchestration
* **LangChain** – prompt and model management
* **Groq (Llama-3.3 70B)** – fast, low-latency inference
* **ChromaDB** – vector database for semantic retrieval
* **Python 3.10+**
