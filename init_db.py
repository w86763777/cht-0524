import argparse
import hashlib
import sys
import uuid
import warnings

import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm


def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help="Path to the csv file.")
    parser.add_argument('--min-words', type=int, default=300,
                        help="The minimum number of words in the content.")
    parser.add_argument('--embedding-model', type=str, default="sentence-transformers/distiluse-base-multilingual-cased-v2",
                        help="HuggingFace embedding model name.")
    parser.add_argument('--db', type=str, default="./db",
                        help="Path to the directory to save the vector database.")
    parser.add_argument('--batch', type=int, default=1024,
                        help="Batch size for querying and adding documents.")
    parser.add_argument('--min-length', type=int, default=100,
                        help="Minimum length of the content.")
    parser.add_argument('--max-length', type=int, default=200,
                        help="Maximum length of the content.")
    args = parser.parse_args()

    df = pd.read_csv(args.data, usecols=['title', 'content', 'source', 'url', 'date'])
    print(f"# news: {len(df):6d}")
    df = df.dropna()
    print(f"# news: {len(df):6d} (dropping NaN)")
    df = df.drop_duplicates('url')
    print(f"# news: {len(df):6d} (dropping duplicates)")
    df = df[df['content'].apply(lambda x: args.min_length <= len(x) <= args.max_length)]
    print(f"# news: {len(df):6d} (Keep content length between {args.min_length} and {args.max_length})")
    df.to_csv(args.data.replace('.csv', '_filtered.csv'), index=False)

    documents = []
    ids = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Build Documents")):
        document = Document(
            page_content=row['title'],
            metadata={
                'content': row['content'],
                'title': row['title'],
                'source': row['source'],
                'url': row['url'],
                'date': row['date'],
                'row': i,
            },
        )
        documents.append(document)
        ids.append(create_uuid_from_string(document.metadata['url']))

    # Disable FutureWarnings from `huggingface_hub`.
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Initialize the embedding function and the database.
    embedding = HuggingFaceEmbeddings(
        model_name=args.embedding_model, model_kwargs={'device': 'cuda:0'})
    db = Chroma(persist_directory=args.db, embedding_function=embedding)

    # Filter out the documents that are already in the database.
    documents_ = []
    ids_ = []
    for st in range(0, len(ids), args.batch):
        batch_ids = ids[st: st + args.batch]
        batch_documents = documents[st: st + args.batch]
        response = db.get(batch_ids)
        response_ids = set(response['ids'])
        for id, document in zip(batch_ids, batch_documents):
            if id not in response_ids:
                documents_.append(document)
                ids_.append(id)

    # splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    # documents_ = splitter.split_documents(documents_)

    # Add the documents to the database.
    documents = documents_
    ids = ids_
    with tqdm(total=len(documents), desc="Build Database") as pbar:
        for st in range(0, len(documents), args.batch):
            batch_texts = [document.page_content for document in documents[st: st + args.batch]]
            batch_metadatas = [document.metadata for document in documents[st: st + args.batch]]
            batch_ids = [id for id in ids[st: st + args.batch]]
            db.add_texts(batch_texts, batch_metadatas, batch_ids)
            pbar.update(len(batch_texts))


if __name__ == '__main__':
    # Fix Chroma's dependency on `sqlite3` module.
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    main()
