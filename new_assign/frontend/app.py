import os
import streamlit as st
import requests
import json
import time
from typing import Generator, Dict, Any
import uuid

# --- Page Configuration ---
st.set_page_config(
    page_title="iPad Assistant",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Clean Professional CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Remove default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] 
    .stAppViewContainer > .main > div {padding-top: 0 !important;}
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }
    
    /* Main container */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Header */
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e1e5e9;
    }
    
    .header h1 {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        font-size: 1rem;
        color: #666666;
        margin: 0;
    }
    
    /* Chat messages */
    .chat-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .user-message {
        background-color: #f0f2f6;
        margin-left: 20%;
        border-left: 3px solid #0066cc;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        margin-right: 20%;
        border-left: 3px solid #28a745;
    }
    
    /* Status indicator */
    .status-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 1rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: #856404;
    }
    
    /* Sources */
    .sources-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        margin-right: 20%;
    }
    
    .sources-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #495057;
    }
    
    .source-link {
        display: block;
        color: #0066cc;
        text-decoration: none;
        font-size: 0.85rem;
        margin: 0.25rem 0;
        padding: 0.25rem 0;
    }
    
    .source-link:hover {
        text-decoration: underline;
    }
    
    /* Input styling */
    .stChatInput > div {
        border: 2px solid #dee2e6;
        border-radius: 8px;
    }
    
    .stChatInput input {
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    
    .stChatInput input:focus {
        border-color: #0066cc !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 102, 204, 0.25) !important;
    }
    
    /* Sidebar */
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    
    .sidebar-title {
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #495057;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #dee2e6;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        border-color: #0066cc;
    }
    
    /* Info boxes */
    .info-item {
        background-color: white;
        border: 1px solid #dee2e6;
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    
    .status-online {
        color: #28a745;
    }
    
    .status-offline {
        color: #dc3545;
    }
    
    /* Error messages */
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 1rem 0;
        margin-right: 20%;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #dee2e6;
        text-align: center;
        color: #666666;
        font-size: 0.9rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .user-message, .assistant-message, .sources-container, .error-message {
            margin-left: 0;
            margin-right: 0;
        }
        
        .main-container {
            padding: 1rem 0.5rem;
        }
        
        .header h1 {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- API Configuration ---
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8003")
STREAM_ENDPOINT = f"{FASTAPI_URL}/ask-stream"
NORMAL_ENDPOINT = f"{FASTAPI_URL}/ask"

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True

# --- Main Layout ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>iPad Assistant</h1>
    <p>Ask questions about Apple iPad models, features, specifications, and more</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)
    
    streaming_enabled = st.toggle(
        "Enable real-time responses", 
        value=st.session_state.streaming_enabled,
        help="Stream responses in real-time"
    )
    st.session_state.streaming_enabled = streaming_enabled
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Actions</div>', unsafe_allow_html=True)
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.success("Conversation cleared")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Status</div>', unsafe_allow_html=True)
    
    # Session info
    if st.session_state.session_id:
        st.markdown(f'<div class="info-item">Session: {st.session_state.session_id[:8]}...</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item">Messages: {len(st.session_state.messages)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-item">No active session</div>', unsafe_allow_html=True)
    
    # API status
    try:
        health_response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.markdown('<div class="info-item status-online">Backend: Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-item status-offline">Backend: Issues</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div class="info-item status-offline">Backend: Offline</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Display Messages ---
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'''
        <div class="chat-message user-message">
            {message["content"]}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="chat-message assistant-message">
            {message["content"]}
        </div>
        ''', unsafe_allow_html=True)
        
        # Show sources
        if "sources" in message and message["sources"]:
            sources_html = '<div class="sources-container">'
            sources_html += '<div class="sources-title">Sources:</div>'
            for i, source in enumerate(message["sources"], 1):
                sources_html += f'<a href="{source}" target="_blank" class="source-link">{i}. {source}</a>'
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

# --- Response Handlers ---
def stream_response(url: str, payload: Dict[Any, Any]) -> Generator[Dict[str, Any], None, None]:
    """Handle streaming response"""
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield {"type": "error", "content": f"Connection error: {str(e)}"}

def handle_streaming_response(prompt: str):
    """Handle streaming response"""
    payload = {"query": prompt}
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id
    
    status_placeholder = st.empty()
    message_placeholder = st.empty()
    sources_placeholder = st.empty()
    
    accumulated_response = ""
    sources = []
    
    try:
        for chunk in stream_response(STREAM_ENDPOINT, payload):
            chunk_type = chunk.get("type")
            chunk_content = chunk.get("content", "")
            
            if chunk_type == "session_id":
                if not st.session_state.session_id:
                    st.session_state.session_id = chunk_content
            
            elif chunk_type == "status":
                status_placeholder.markdown(f'''
                <div class="status-message">
                    {chunk_content}
                </div>
                ''', unsafe_allow_html=True)
            
            elif chunk_type == "sentence":
                accumulated_response += chunk_content + " "
                message_placeholder.markdown(f'''
                <div class="chat-message assistant-message">
                    {accumulated_response}
                </div>
                ''', unsafe_allow_html=True)
            
            elif chunk_type == "sources":
                sources = chunk_content
                if sources:
                    sources_html = '<div class="sources-container">'
                    sources_html += '<div class="sources-title">Sources:</div>'
                    for i, source in enumerate(sources, 1):
                        sources_html += f'<a href="{source}" target="_blank" class="source-link">{i}. {source}</a>'
                    sources_html += '</div>'
                    sources_placeholder.markdown(sources_html, unsafe_allow_html=True)
            
            elif chunk_type == "complete":
                status_placeholder.empty()
                message_placeholder.markdown(f'''
                <div class="chat-message assistant-message">
                    {accumulated_response.strip()}
                </div>
                ''', unsafe_allow_html=True)
                break
            
            elif chunk_type == "error":
                status_placeholder.empty()
                message_placeholder.markdown(f'''
                <div class="error-message">
                    {chunk_content}
                </div>
                ''', unsafe_allow_html=True)
                break
        
        # Store response
        if accumulated_response.strip():
            assistant_message = {
                "role": "assistant", 
                "content": accumulated_response.strip(),
                "sources": sources
            }
            st.session_state.messages.append(assistant_message)
                
    except Exception as e:
        status_placeholder.empty()
        message_placeholder.markdown(f'''
        <div class="error-message">
            Error: {str(e)}
        </div>
        ''', unsafe_allow_html=True)

def handle_normal_response(prompt: str):
    """Handle normal response"""
    message_placeholder = st.empty()
    message_placeholder.markdown('''
    <div class="status-message">
        Processing your request...
    </div>
    ''', unsafe_allow_html=True)

    try:
        payload = {"query": prompt}
        if st.session_state.session_id:
            payload["session_id"] = st.session_state.session_id

        response = requests.post(NORMAL_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        full_response = response.json()
        
        answer = full_response.get("answer", "Sorry, I couldn't get a response.")
        message_placeholder.markdown(f'''
        <div class="chat-message assistant-message">
            {answer}
        </div>
        ''', unsafe_allow_html=True)

        if "session_id" in full_response and st.session_state.session_id is None:
            st.session_state.session_id = full_response["session_id"]

        assistant_message = {
            "role": "assistant", 
            "content": answer,
            "sources": full_response.get("sources", [])
        }
        st.session_state.messages.append(assistant_message)

        # Display sources
        sources = full_response.get("sources", [])
        if sources:
            sources_html = '<div class="sources-container">'
            sources_html += '<div class="sources-title">Sources:</div>'
            for i, source in enumerate(sources, 1):
                sources_html += f'<a href="{source}" target="_blank" class="source-link">{i}. {source}</a>'
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

    except requests.exceptions.RequestException as e:
        error_message = f"Could not connect to the backend: {e}"
        message_placeholder.markdown(f'''
        <div class="error-message">
            {error_message}
        </div>
        ''', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Chat Input ---
if prompt := st.chat_input("Ask me about iPad features, models, pricing, or specifications..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'''
    <div class="chat-message user-message">
        {prompt}
    </div>
    ''', unsafe_allow_html=True)

    # Process response
    if st.session_state.streaming_enabled:
        handle_streaming_response(prompt)
    else:
        handle_normal_response(prompt)

# --- Footer ---
st.markdown(f"""
<div class="footer">
    <strong>iPad Assistant</strong><br>
    Streaming: {'Enabled' if st.session_state.streaming_enabled else 'Disabled'} | 
    Session: {st.session_state.session_id[:8] + '...' if st.session_state.session_id else 'None'} | 
    Messages: {len(st.session_state.messages)}
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)