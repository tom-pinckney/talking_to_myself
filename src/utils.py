from typing import List
from scipy import spatial

def chat_message(question, context, messages, client, model="gpt-4-turbo", stream=False):
    """
    Wrapper function to send a new message and append response to the chat session
    """

    messages.append({"role": "user", "content": f"Context: {context} \n\n---\n\nQuestion: {question}\nAnswer:"})
    
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=450,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stream=stream,
        ).choices[0].message.content

    messages.append({"role": "assistant", "content": response})

    return messages

def print_response(messages, role='assistant'):
    """
    Takes chat message session and prints the last message for the given role
    """
    for i in range(1,len(messages)):
        i = i*-1
        message = messages[i]
        if message['role']==role:
            print(message['content'])
            return None
    print("No message with provided role")


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def get_embedding(text, client):
    """
    Embed string, can be used in apply pandas call
    """

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    # Returns embedding vector
    return response.data[0].embedding

# Update function to not have hard coded usernames
def create_context(
    question, df, client, max_len=1800
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = get_embedding(question, client)

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["context_chunk"])

    # Return the context
    return "\n\n###\n\n".join(returns)
