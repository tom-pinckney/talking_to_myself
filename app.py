from openai import OpenAI
import streamlit as st
import pandas as pd
from src.utils import create_context
import ast
import re
import os
import json
from datetime import datetime as dt

session_key = dt.today().strftime("%Y-%m-%d%H-%M-%S")

USERNAME = "Thomas Pinckney"
SYSTEM_MESSAGE = {"role": "system", "content": f"You are {USERNAME},a teenage boy. Use the given context to respond in the voice of {USERNAME}. Respond openly in a few sentences giving details about your life. Do not ask questions just be detailed about yourself."}


@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    data = pd.read_csv(url)
    data['embeddings'] = data.embeddings.apply(ast.literal_eval)
    return data


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

def clean_user_messages(s):
    s = remove_text_between(s)
    s = remove_answer_from_end(s)
    return s

df = load_data('data/embedded_emails.csv')


st.title("Talking to Myself")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_MESSAGE]

for message in st.session_state.messages:
    if message['role'] != "system":
        with st.chat_message(message["role"]):
            message_to_print = clean_user_messages(message["content"])
            st.markdown(message_to_print)

if prompt := st.chat_input("What is your name?"):
    context = create_context(prompt, df, client, max_len=750)
    st.session_state.messages.append({"role": "user", "content": f"Context: {context} \n\n---\n\nQuestion: {prompt}\nAnswer:"})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Remove prior context from messages
        messages = [
                {"role": m["role"], "content": clean_user_messages(m["content"])}
                for m in st.session_state.messages
            ]
        # Replace the latest message with the full context
        messages[-1] = st.session_state.messages[-1]
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=messages,
            temperature=0.5,
            max_tokens=450,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Specify the subfolder and file name
    subfolder = 'conversations'
    filename = f'{session_key}_output.json'
    path = os.path.join(subfolder, filename)

    # Create the subfolder if it doesn't exist
    os.makedirs(subfolder, exist_ok=True)
    with open(path, 'w') as file:
        json.dump(st.session_state.messages, file, indent=4)