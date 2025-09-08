import streamlit as st
import pandas as pd
from agent_llm import LLMDataAgent

st.set_page_config(page_title="LLM Conversational Data Agent", layout="wide")
st.title("💬 Conversational Data Cleaning Agent (LLM-powered)")

# Session state for chat
if 'chat' not in st.session_state:
    st.session_state.chat = []

if 'agent' not in st.session_state:
    st.session_state.agent = None

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file and st.session_state.agent is None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.session_state.agent = LLMDataAgent(df)
    st.session_state.chat.append(("agent", f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}"))
    st.experimental_rerun()

# Display chat
for sender, message in st.session_state.chat:
    if sender == 'agent':
        st.markdown(f"**Agent:** {message}")
    else:
        st.markdown(f"**You:** {message}")

# User input
if st.session_state.agent is not None:
    user_input = st.text_input("Type your message:", key="user_input")
    if user_input:
        st.session_state.chat.append(("user", user_input))
        response = st.session_state.agent.chat(user_input)
        st.session_state.chat.append(("agent", response))
        st.experimental_rerun()

# Download buttons after cleaning
import os
if os.path.exists('cleaned_dataset.csv'):
    st.download_button(
        label="📥 Download Cleaned Dataset",
        data=open('cleaned_dataset.csv', 'rb').read(),
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )
    st.download_button(
        label="📥 Download Cleaning Report",
        data=open('report.csv', 'rb').read(),
        file_name="report.csv",
        mime="text/csv"
    )
