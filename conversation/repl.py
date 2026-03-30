from identity.models import User
from conversation.guards import validate_input, extract_structured


def run(agent, user: User) -> None:
    """Run the interactive REPL loop."""
    print(f"\nReady! Logged in as {user.name} ({user.department}, {user.permission_level.name} access).")
    print("Ask questions about your department's documents (type 'quit' to exit).\n")

    while True:
        query = input("You: ").strip()

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        if not validate_input(query):
            print("Input validation failed: Query must be between 10 and 200 characters.")
            continue

        full_response = ""
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            msg = step["messages"][-1]
            msg.pretty_print()
            full_response = msg.content

        success, result = extract_structured(full_response)
        if success:
            print(f"\n[Policy Summary] {result}")

        print()
