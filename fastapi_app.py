# A fastAPI app to insert queries which are then embedded by text-ada-002
# and then queries pinecone index to return the most similar documents.
# Also include querying by metadata, namely date range.

import os
from dataclasses import dataclass
from datetime import date

import pinecone
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings

# load environment variables
load_dotenv()

# initialise pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")


def load_xlsx_files(data_dir="./data/xlsx"):
    df_list = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_excel(file_path)
            df_list.append(df)

    combined_df = pd.concat(df_list)
    combined_df["brief"].fillna("", inplace=True)
    combined_df["headline"].fillna("", inplace=True)

    combined_df["text"] = (
        "Headline:\n" + combined_df["headline"] + "\n" + combined_df["brief"]
    )
    combined_df["id"] = combined_df["id"].astype(str)

    data_dict = combined_df.set_index("id")["text"].to_dict()

    return data_dict


@dataclass
class VectorMatch:
    id: str
    ordinal: date
    score: float
    text: str

    def __init__(self, id: str, ordinal: float, score: float, text: str):
        # parse data values
        self.id = id
        self.ordinal = date.fromordinal(int(ordinal))
        self.score = score
        self.text = text


app = FastAPI()

# Define a list of allowed origins
# You should replace this with the origin(s) of your Vue application
origins = [
    "http://localhost",
    "http://localhost:8080",
]

# Add the CORS middleware to your FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DICT = load_xlsx_files(data_dir="./data/xlsx")


@app.post("/query")
def query(query: str, date_from: date = None, date_to: date = None):
    # get vector embedding from text query
    query_embedding = embeddings_model.embed_query(query)

    # check if pinecone index exists
    assert os.getenv("PINECONE_INDEX") in pinecone.list_indexes()
    pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX"))

    # create filter
    filter = None
    if date_from and date_to:
        filter = {
            "ordinal": {
                "$gte": date_from.toordinal() if date_from else None,
                "$lte": date_to.toordinal() if date_to else None,
            }
        }

    # query pinecone index
    results = pinecone_index.query(
        query_embedding,
        top_k=10,
        filter=filter,
        include_metadata=True,
    )["matches"]
    # order results by score; lowest first
    results = sorted(results, key=lambda x: x["score"])

    # creates VectorMatch objects
    matches = [
        VectorMatch(
            id=r["id"],
            ordinal=r["metadata"]["ordinal"],
            score=r["score"],
            text=DATA_DICT[r["id"]] if r["id"] in DATA_DICT else None,
        )
        for r in results
    ]

    return {"results": matches}
