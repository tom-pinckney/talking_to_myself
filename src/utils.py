from scipy.spatial.distance import cdist

def semantic_search(query_embedding, doc_embeddings, doc_ids, top_k=5):
    # Calculate cosine distances from the query to all documents
    distances = cdist([query_embedding], doc_embeddings, 'cosine')[0]

    # Get the top K smallest distances
    top_k_indices = distances.argsort()[:top_k]

    # Return the corresponding document IDs and their distances
    return [(doc_ids[idx], distances[idx]) for idx in top_k_indices]

# Update function to not have hard coded usernames
def create_context(
    question, df, embeddings, embedding_model, username="Kay", max_len=1800
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = embedding_model.encode(question)

    filter = {
      'Kay': "kayharper09@gmail.com",
      "Thomas": "pressingon1617@gmail.com"
    }

    email = filter[username]

    # Filter the DataFrame to only include messages from the specified email address
    filtered_df = df[df['from'].str.contains(email)]
    filtered_embeddings = embeddings[filtered_df.index.to_list(), :]

    # Get the distances from the embeddings
    filtered_df['distances'] = cdist([q_embeddings], filtered_embeddings, 'cosine')[0]


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in filtered_df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)