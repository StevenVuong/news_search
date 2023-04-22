import os
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pinecone
from dotenv import load_dotenv
import asyncio


# load environment variables
load_dotenv()

# initialise pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")


def embed_text(text):
    return embeddings_model.embed_query(text)


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

    for file_path in xlsx_dir.glob("*.xlsx"):
        # read xlsx file
        df = pd.read_excel(file_path)

        # create text to embed
        df["text"] = "Headline:\n" + df["headline"] + "\n" + df["brief"]

        # go from datetime to date; and get ordinal time
        df["date"] = pd.to_datetime(df["updateTime"]).dt.date
        df["ordinal"] = df["date"].apply(lambda x: x.toordinal())

        # drop unwanted columns
        df.drop(["headline", "brief", "updateTime", "date"], axis=1, inplace=True)

        # just do head for now; delete this line later
        df = df.head(2)
        print("-" * 80)
        print(df.head()))

        # loop through and embed queries; no support for documents at the moment
        with ThreadPoolExecutor() as executor:
            embedded_texts = list(executor.map(embed_text, df["text"].values))

        # create pinecone list to insert into index of [("id", [vector], {"ordinal":})]
        pinecone_upsert_list = list(
            zip(
                df["id"].tolist(),
                embedded_texts,
                [{"ordinal": x} for x in df["ordinal"].tolist()],
            )
        )

        # upsert pinecone index
        pinecone_index.upsert(pinecone_upsert_list)

        # Get index statistics
        stats = pinecone_index.describe_index_stats()
        print("-" * 80)
        print("Index statistics:")
        print(stats)
        break
