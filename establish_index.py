# Creates a pinecone index and populates with embedded text from.
# xlsx files in the data/xlsx folder. Embedding is with OpenAI text-embedding-ada-002 model.

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pinecone
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm import tqdm

# load environment variables
load_dotenv()

# initialise pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
# feed model one line at a time; very slow but helps us resolve a particular bug..
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)


def embed_text(text):
    try:
        return embeddings_model.embed_query(text)
    except:
        raise LookupError(f"Error embedding text: {text}")


def embed_text_with_delay(text):
    time.sleep(0.9)  # add a 0.9 second delay
    return embed_text(text)


if __name__ == "__main__":
    # check if pinecone index exists
    if os.getenv("PINECONE_INDEX") not in pinecone.list_indexes():
        # create pinecone index. Note: can take a few mins to create
        pinecone.create_index(
            os.getenv("PINECONE_INDEX"), dimension=1536, metric="cosine", pod_type="p1"
        )
    pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX"))

    # load xlsx files
    xlsx_dir = Path("./data/xlsx")

    # loop through xlsx files
    for file_path in xlsx_dir.glob("*.xlsx"):
        print(f"Processing {file_path}...")

        # read xlsx file
        df = pd.read_excel(file_path)

        # clean up data
        df["brief"].fillna("", inplace=True)
        df["headline"].fillna("", inplace=True)
        # create text to embed
        df["text"] = "Headline:\n" + df["headline"] + "\n" + df["brief"]

        # go from datetime to date; and get ordinal time
        df["date"] = pd.to_datetime(df["updateTime"]).dt.date
        df["ordinal"] = df["date"].apply(lambda x: x.toordinal())

        # drop unwanted columns
        df.drop(["headline", "brief", "updateTime", "date"], axis=1, inplace=True)

        # set asset type for remaining columns
        df["text"] = df["text"].astype(str)
        df["ordinal"] = df["ordinal"].astype(int)
        df["id"] = df["id"].astype(str)

        # just do head for now; delete this line later
        print("-" * 80)
        print(f"Embedding {len(df)} texts...")

        # pinecone has upsert limit of 2MB and cannot be >1000 vectors.
        batch_size = 100
        # batch upsert
        for i in range(0, len(df), batch_size):
            print(f"Batch {i} to {i + batch_size}...")
            df_batch = df.iloc[i : i + batch_size]

            # loop through and embed queries; no support for documents at the moment
            with ThreadPoolExecutor(max_workers=5) as executor:
                embedded_texts = list(
                    tqdm(executor.map(embed_text_with_delay, df_batch["text"].values))
                )

            # create pinecone list to insert into index of [("id", [vector], {"ordinal":})]
            pinecone_upsert_list = list(
                zip(
                    df_batch["id"].tolist(),
                    embedded_texts,
                    [{"ordinal": x} for x in df_batch["ordinal"].tolist()],
                )
            )

            # upsert pinecone index
            pinecone_index.upsert(pinecone_upsert_list)

            # Get index statistics
            stats = pinecone_index.describe_index_stats()
            print("-" * 80)
            print("Index statistics:")
            print(stats)
