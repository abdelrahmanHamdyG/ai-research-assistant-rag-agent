from src.langgraph_workflow.graph import create_graph

compiled_graph = create_graph()

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
