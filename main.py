from dotenv import load_dotenv
import streamlit as st

from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

st.title("Ask Chatty Bot")
st.markdown("Ask Chatty Bot anything you want and it will answer your questions.")

llm = GoogleGenerativeAI(model="gemini-3.6-flash")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

# Get user input
query = st.chat_input("Ask anything...")

if query:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message in chat
    st.chat_message("user").markdown(query)

    # Generate response from the model
    response = llm.invoke(query)
    st.chat_message("ai").markdown(response)
    st.session_state.messages.append({
        "role": "ai",
        "content": response
    })