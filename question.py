import openai
import os
import pinecone
from IPython.display import Markdown

embed_model = 'text-embedding-ada-002'

# system message to 'prime' the model
primer = f"""You are an enthusiastic, highly intelligent support professional at Atomic Jolt.
Answer user questions based on the information provided by the user above each question.
Don't specifically include Utah State. Do output markdown, html, links to documents and image tags.
If the information can not be found in the information provided by the user you truthfully say "I don't know".
"""

# Initialize Pinecone
def init_index():
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENV")
    )

    index_name = "atomic-assessments-docs"

    # connect to index
    index = pinecone.GRPCIndex(index_name)

    return index

def augmented_query(index, query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts (including the questions)
    res = index.query(xq, top_k=5, include_metadata=True)

    # get list of retrieved text
    contexts = [item['metadata']['text'] for item in res['matches']]
    return "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

def query(query):
    index = init_index()
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query(index, query)}
        ]
    )


    # Response example:
    # {
    # "choices": [
    #     {
    #     "finish_reason": "stop",
    #     "index": 0,
    #     "message": {
    #         "content": "To set time extensions or extra attempts for one or several students in Atomic Assessments, you can navigate to the \"Assign\" section in the assessment \"Settings\" tab. In this section, set the general number of allowed attempts, availability date, and time limit. To add accommodations for one or more students, select \"Add Override\". Add the alternate settings and select \"Save\".",
    #         "role": "assistant"
    #     }
    #     }
    # ],
    # "created": 1679967528,
    # "id": "chatcmpl-6ysaOWEwvYcHXxdDNo1YQBg525USi",
    # "model": "gpt-3.5-turbo-0301",
    # "object": "chat.completion",
    # "usage": {
    #     "completion_tokens": 76,
    #     "prompt_tokens": 2153,
    #     "total_tokens": 2229
    # }
    # }
    return res['choices'][0]['message']['content']


# response = query("Tell me about Atomic Assessments.")
# response = query("How do I setup student specific overrides using Atomic Assessments?")
# response = query("Help me setup a multiple choice question in Atomic Assessments.")
# response = query("How do I build a template in Atomic Assessments that I can reuse for future assessment?")
response = query("How do I use Atomic Assessments?")
print(response)