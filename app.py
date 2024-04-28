from openai import OpenAI
import streamlit as st
import pandas as pd
from src.utils import create_context
import ast
import re

USERNAME = "Thomas Pinckney"
SYSTEM_MESSAGE = {"role": "system", "content": f"You are {USERNAME}, a teenage boy: Answer the chat messages pretending to be {USERNAME}, using the given context from {USERNAME}'s emails. Write in the same style as the context."}


@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    df['embeddings'] = df.embeddings.apply(ast.literal_eval)
    return df


def remove_text_between(s, start_marker="Context:", end_marker="Question:"):
    """
    Remove the context for printing out messages
    :param s: string
    :param start_marker:
    :param end_marker:
    :return: the string with everything between start marker and end marker removed
    """
    pattern = re.escape(start_marker) + ".*?" + re.escape(end_marker)
    cleaned_string = re.sub(pattern, '', s, flags=re.DOTALL)
    return cleaned_string

def remove_answer_from_end(s):
    pattern = r"\s*Answer:\s*$"
    cleaned_string = re.sub(pattern, '', s)
    return cleaned_string


df = load_data('data/embedded_emails.csv')


st.title("Talking to Myself")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_MESSAGE]

for message in st.session_state.messages:
    if message['role'] != "system":
        with st.chat_message(message["role"]):
            message_to_print = remove_text_between(message["content"])
            message_to_print = remove_answer_from_end(message_to_print)
            st.markdown(message_to_print)

if prompt := st.chat_input("What is your name?"):
    context = create_context(prompt, df, client)
    st.session_state.messages.append({"role": "user", "content": f"Context: {context} \n\n---\n\nQuestion: {prompt}\nAnswer:"})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=0,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})