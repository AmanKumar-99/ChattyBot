from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_groq import ChatGroq
from langchain_exa import ExaSearchResults
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b", max_tokens=512)
search = ExaSearchResults()
tools = [search]

# Initialize memory in session state if not already present
# This ensures that the memory persists across user interactions in the Streamlit app.
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

# Create an agent with the specified model, tools, and memory.
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are a helpful assistant that can answer questions based on the provided context.",
)

# Building Web Interface.
st.subheader("Ask Chatty Bot")
st.markdown("Ask Chatty Bot anything you want and it will answer your questions.")

query = st.chat_input("Ask here...")

# Display chat messages from history on app rerun
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

# Handle user query and generate response
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    response = agent.stream(
        {
            "messages": [{"role": "user", "content": query}],
        },
        {
            "configurable":{ "thread_id": "1"}
        },
        stream_mode="messages"
    )

    # Display AI response in real-time as it is generated
    ai_container = st.chat_message("ai")
    # Use a container to display the AI's response incrementally
    with ai_container:
        space = st.empty()

        msg = ""

        for chunk in response:
            msg = msg + chunk[0].content
            space.write(msg)

        st.session_state.history.append({"role": "ai", "content": msg})