import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Page Configuration
st.set_page_config(page_title="OpoQ Chatbot", page_icon="💬", layout="centered")

st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom, #1e293b, #0f172a);
            color: #f8fafc;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 800px;
            margin: auto;
        }
        h1 {
            color: #f1f5f9;
            font-weight: 700;
            text-align: center;
        }
        p {
            color: #cbd5e1;
            text-align: center;
            font-size: 18px;
            margin-top: -10px;
        }
        .stTextArea textarea {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px !important;
            background-color: #f1f5f9;
            color: #0f172a;
            border-radius: 10px;
            padding: 10px;
            border: none;
        }
        .stTextInput > div > div > input {
            background-color: #f1f5f9;
            color: #0f172a;
            border-radius: 8px;
            padding: 8px;
        }
        .stButton button {
            font-weight: 600;
            border-radius: 10px;
            padding: 10px 24px;
            background: linear-gradient(to right, #14b8a6, #0ea5e9);
            color: white;
            border: none;
            transition: 0.3s ease-in-out;
        }
        .stButton button:hover {
            background: linear-gradient(to right, #0d9488, #0284c7);
        }
        .stSidebar {
            background-color: #1e293b;
        }
        .stSidebar > div {
            padding: 20px;
            color: #e2e8f0;
        }
        .stSidebar .stTextInput input {
            background-color: #f1f5f9;
            color: #1e293b;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>💬 OpoQ Chatbot</h1><p>Select your preferred LLM provider and begin your conversation with OpoQ</p>", unsafe_allow_html=True)


# State Initialization
for key in ['conversation', 'messages', 'api_key', 'provider']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'conversation' else ([] if key == 'messages' else '')

# Sidebar Controls
st.sidebar.header("🛠️ Chat Settings")

provider_option = st.sidebar.selectbox("LLM Provider", ["Select", "OpenAI", "Groq"])
st.session_state['provider'] = provider_option
st.session_state['api_key'] = st.sidebar.text_input("API Key", type="password")
submit_api = st.sidebar.button("🔌 Connect")

# LLM Initialization
def initialize_llm(provider, api_key):
    try:
        if provider == "OpenAI":
            llm = OpenAI(
                temperature=0,
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo-instruct"
            )
        elif provider == "Groq":
            llm = ChatGroq(
                temperature=0,
                groq_api_key=api_key,
                model_name="llama3-70b-8192"
            )
        else:
            return None

        _ = llm.invoke("Hello")  # Validate key
        return llm

    except Exception as e:
        st.error(f"API Key validation failed: {str(e)}")
        return None

# Conversation Summarizer
def summarize():
    if st.session_state['conversation']:
        return st.session_state['conversation'].memory.buffer
    return "No conversation to summarize."

# Connect to LLM
if submit_api and st.session_state['api_key'] and st.session_state['provider'] != "Select":
    with st.spinner("Connecting to provider..."):
        llm = initialize_llm(st.session_state['provider'], st.session_state['api_key'])
        if llm:
            st.session_state['conversation'] = ConversationChain(
                llm=llm,
                verbose=False,
                memory=ConversationSummaryMemory(llm=llm)
            )
            st.success(f"{provider_option} connected successfully.")
        else:
            st.session_state['conversation'] = None

# Chat UI
if st.session_state['conversation']:
    if st.sidebar.button("📄 Summarize Chat"):
        st.sidebar.subheader("🧾 Chat Summary")
        st.sidebar.write(summarize())

    with st.container():
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_area("💬 Your Message", key='input', height=100)
            send = st.form_submit_button("➡️ Send")

        if send and user_input:
            st.session_state['messages'].append(user_input)
            with st.spinner("🤖 Generating response..."):
                response = st.session_state['conversation'].predict(input=user_input)
            st.session_state['messages'].append(response)

    # Display messages with chat styling
    for i, msg in enumerate(st.session_state['messages']):
        is_user = i % 2 == 0
        message(
            msg,
            is_user=is_user,
            key=f"{i}_{'user' if is_user else 'bot'}",
            avatar_style="thumbs" if is_user else "bottts"
        )
else:
    st.info("👈 Please select a provider and enter a valid API key to begin chatting.")
