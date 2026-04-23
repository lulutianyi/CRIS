import streamlit as st
import requests

API_URL = "https://api.edgefn.net/v1/chat/completions"
API_KEY = "sk-F5uBEnH4Pag26IaE0209244cE9B54049BcE0B25f7f994415"

st.set_page_config(page_title="My LLM Chat", layout="wide")

st.title("💬 My LLM Chatbot")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 调 API
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "DeepSeek-R1-0528",
        "messages": st.session_state.messages,
        "stream": False   # 先关掉流式，后面再开
    }

    response = requests.post(API_URL, headers=headers, json=data)
    reply = response.json()["choices"][0]["message"]["content"]

    # 显示回复
    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})