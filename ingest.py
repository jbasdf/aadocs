import os
import requests
import string
import tempfile
import tiktoken
import pinecone
import urllib

from bs4 import BeautifulSoup
from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embed_model = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    document_model_name=embed_model,
    query_model_name=embed_model
)

# create the length function
tokenizer = tiktoken.get_encoding('cl100k_base')
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

def url_to_filename(url):
    # Extract the file name from the URL
    parsed_url = urllib.parse.urlparse(url)
    file_name = os.path.basename(parsed_url.path)

    # Remove any invalid characters from the file name
    valid_chars = '-_.() %s%s' % (string.ascii_letters, string.digits)
    file_name = ''.join(c for c in file_name if c in valid_chars)

    # Truncate the file name if it's too long
    max_len = 255
    if len(file_name) > max_len:
        file_name = file_name[:max_len]

    return file_name

def embed_it(url, text, index):
    filename = url_to_filename(url)

    # Create a temporary file and write the html to it
    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".html", prefix=filename) as f:
        f.write(text)
        f.flush()  # Ensure the contents are written to disk

        loader = BSHTMLLoader(f.name)
        document = loader.load()[0]

    data = document.page_content.strip()
    title = document.metadata.get('title')

    # first get metadata fields for this record
    metadata = {
        'id': url,
        'source': url,
        'title': title
    }

    # now we create chunks from the record text
    record_texts = text_splitter.split_text(data)

    # create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]

    print("ingesting chunks " + str(len(record_texts)) + url)

    # if we have reached the batch_limit we can add texts
    ids = [url+str(i) for i in range(len(record_texts))]
    embeds = embed.embed_documents(record_texts)
    index.upsert(vectors=zip(ids, embeds, record_metadatas))

def load(index, url, base_url, test_url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    embed_it(url, res.text, index)

    local_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urllib.parse.urljoin(base_url, href)
        if full_url.startswith(test_url):
            local_links.append(full_url)

    return local_links


def crawl(index, base_url, test_url):
    links = load(index, base_url, base_url, test_url)

    all_links = set(links)

    for link in links:
        sub_links = load(index, link, base_url, test_url)
        all_links.update(sub_links)

    return all_links

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

def main():
    links = []
    index = init_index()

    base_url = "https://support.atomicjolt.com/support/solutions/22000039715"
    test_url = "https://support.atomicjolt.com/support/"
    links.extend(crawl(index, base_url, test_url))

    base_url = "https://www.usu.edu/teach/help-topics/teaching-softwares/atomic-assessments/"
    test_url = "https://www.usu.edu/teach/help-topics/teaching-softwares/atomic-assessments/"
    links.extend(crawl(index, base_url, test_url))

    # for link in links:
    #     print(link)

    print(index.describe_index_stats())


main()
