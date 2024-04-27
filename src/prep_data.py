import mailbox
import pandas as pd
import hashlib
from openai import OpenAI
from email.utils import parsedate_to_datetime
import tiktoken
import numpy as np
import streamlit as st

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TARGET_EMAIL = st.secrets["TARGET_EMAIL"]
client = OpenAI(api_key=OPENAI_API_KEY)

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

def extract_recent_message(body):
    # Split the body into lines
    lines = body.splitlines()
    # Keep only lines that do not start with '>'
    recent_lines = [line for line in lines if not line.strip().startswith('>')]
    # Join the lines back into a single string
    # Drop the last line it is either blank or the header of the prior email
    return "\n".join(recent_lines[:-1])


def extract_email_data_to_dataframe(mbox_path):
    # Open the mbox file
    mbox = mailbox.mbox(mbox_path)
    emails = []

    # Process each message in the mbox
    for message in mbox:
        if message.is_multipart():
            body = ''.join(part.get_payload(decode=True).decode('utf-8', errors='ignore')
                           for part in message.get_payload() if part.get_content_type() == 'text/plain')
        else:
            body = message.get_payload(decode=True).decode('utf-8', errors='ignore')

        # Extract fields
        date = parsedate_to_datetime(message['date']).isoformat() if message['date'] else 'Unknown'
        sender = message['from']
        recipients = message['to']
        # Generate hash ID from email text
        email_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()[:24]

        # Append to list as a dictionary
        emails.append({
            'id': email_hash,
            'text': extract_recent_message(body),
            'time': date,
            'from': sender,
            'to': recipients
        })

    # Convert list to DataFrame
    df = pd.DataFrame(emails)
    return df

def split_text_into_chunks(df, max_tokens):
    # Prepare a container for the new DataFrame rows
    new_rows = []

    # Iterate over each row in the DataFrame
    for index_val, row in df.iterrows():
        if row['n_tokens'] > max_tokens:
            # Calculate number of chunks needed
            num_chunks = np.ceil(row['n_tokens'] / max_tokens).astype(int)
            words = row['text'].split()

            # Determine approximately equal sizes for each chunk
            chunk_size = len(words) // num_chunks
            extra = len(words) % num_chunks

            # Create each chunk
            start = 0
            for i in range(num_chunks):
                # Adjust the last chunk to take any remaining words
                end = start + chunk_size + (1 if i < extra else 0)
                chunk_text = ' '.join(words[start:end])
                new_rows.append({'email_id': index_val, 'chunk_id': f"{index_val}-{i+1}", 'text': chunk_text, 'n_tokens': len(chunk_text.split()), "time": row['time'], 'from':row['from'], 'to':row['to']})
                start = end
        else:
            # If no splitting is needed, keep the row as it is
            row['chunk_id'] = index_val
            row['email_id'] = index_val
            new_rows.append(row.to_dict())

    # Create a new DataFrame from the list of new rows
    new_df = pd.DataFrame(new_rows)
    return new_df

def preprocess_data(df, max_tokens):
  df.set_index('id', inplace=True)
  df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
  df = split_text_into_chunks(df, max_tokens)
  return df


def get_embedding(text):
    """
    Embed string, can be used in apply pandas call
    """

    response = client.embeddings.create(
        input=df_emails.text.values[0],
        model="text-embedding-3-small"
    )

    # Returns embedding vector
    return response.data[0].embedding


# Specify the path to your mbox file
df_emails = extract_email_data_to_dataframe('../Data/GptDump.mbox')
df_emails = preprocess_data(df_emails, 100)
df_emails = df_emails.loc[df_emails['from'].str.contains(TARGET_EMAIL)].reset_index(drop=True)
df_emails['embeddings'] = df_emails['text'].apply(get_embedding)
df_emails.to_csv('../data/embedded_emails.csv', index=False)