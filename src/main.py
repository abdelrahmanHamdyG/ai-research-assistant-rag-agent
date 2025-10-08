from src.langgraph_workflow.graph import create_graph
import os
import json
from datetime import datetime, timedelta
from src.vector_db.query_embeddings import remove_old_papers
from src.vector_db.store_embedding import store_papers_embedding
from src.ingestion.fetch_openalex import fetch_papers
from src.ingestion.preprocess import preprocess_papers

compiled_graph = create_graph()

import os
import json
from datetime import datetime, timedelta

import os
import json
from datetime import datetime, timedelta

kind = [True, True]

# ---- 1️⃣ Check for the base chunks file ----
if os.path.exists("data/processed/chunks.jsonl"):
    kind[0] = False

# ---- 2️⃣ Check for recent chunks freshness ----
recent_path = "data/processed/chunks_recent.jsonl"
metadata_recent_path = "data/processed/metadata_recent.json"

if os.path.exists(recent_path):
    try:
        # Load the last line (most recent entry)
        with open(recent_path, "r", encoding="utf-8") as f:
            last_line = None
            for line in f:
                last_line = line.strip()
        
        if last_line:
            data = json.loads(last_line)
            last_date_str = data.get("date_published")
            
            if last_date_str:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                today = datetime.now().date()
                days_since_last = (today - last_date).days

                # If last paper is older than 3 days → rebuild needed
                if days_since_last > 3:
                    # print("it's")
                    kind[1] = True
                    # Remove old files
                    os.remove(recent_path)
                    if os.path.exists(metadata_recent_path):
                        os.remove(metadata_recent_path)
                    remove_old_papers()
                    
                else:
                    kind[1] = False
    except Exception as e:
        print(f"⚠️ Error checking recency: {e}")
        kind[1] = True
else:
    kind[1] = True

print("kind =", kind)




if kind[0] or kind[1]:

    fetch_papers(kind)
    preprocess_papers(kind)
    store_papers_embedding([True,True])







# --- 1️⃣ First user prompt
query = input("Enter your prompt:\n").strip()

state = {
    "query":query,
    "chat_history": [{"role": "user", "content": query}],
    "intent_info": None,
    "period_papers": [],
    "last_bot_response": None
}

# --- 2️⃣ Conversation loop
while True:
    # Call the graph with the current state
    output = compiled_graph.invoke(state)

    # Get the bot's reply (assuming graph puts bot messages in chat_history)
    bot_reply = output["chat_history"][-1]["content"]
    print("Bot:", bot_reply)

    # Prepare for next round
    next_query = input("\nEnter your next prompt (or type 'exit' to quit):\n").strip()
    if next_query.lower() == "exit":
        break

    # Append the user message
    output["chat_history"].append({"role": "user", "content": next_query})

    # ✅ Keep only the last 3 messages in history
    output["chat_history"] = output["chat_history"][-3:]

    # The new state for the next iteration
    state = output
