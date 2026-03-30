from identity.models import User
from conversation.guards import validate_input, extract_structured


def run(agent, user: User) -> None:
    """Run the interactive REPL loop."""
    print(f"\nReady! Logged in as {user.name} ({user.department}, {user.permission_level.name} access).")
    print("Ask questions about your department's documents (type 'quit' to exit).")
    print("Prefix with 'extract:' to extract structured opening data.\n")

    while True:
        query = input("You: ").strip()

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        if not validate_input(query):
            print("Input validation failed: Query must be between 10 and 200 characters.")
            continue

        extract_mode = query.lower().startswith("extract:")
        if extract_mode:
            query = query[len("extract:"):].strip()
            query += (
                "\n\nRespond with ONLY a JSON object matching this schema, no other text:"
                '\n{"name": "<opening name>", "moves": "<moves in standard algebraic notation>"}'
            )

        full_response = ""
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            msg = step["messages"][-1]
            msg.pretty_print()
            full_response = msg.content

        if extract_mode:
            success, result = extract_structured(full_response)
            if success:
                print(f"\n[Extracted Opening] {result}")
            else:
                print(f"\n[Extraction failed] Could not extract structured opening data.")
                print(f"Error: {result}")

        print()
