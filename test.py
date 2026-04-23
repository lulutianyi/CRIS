import streamlit as st

st.title("测试页面")
st.write("如果你能看到这句话，说明 Streamlit 渲染正常！")

user_input = st.text_input("随便输入点什么")
if user_input:
    st.write(f"你输入了: {user_input}")