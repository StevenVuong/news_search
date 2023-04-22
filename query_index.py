# A fastAPI app to insert queries which are then embedded by text-ada-002
# and then queries pinecone index to return the most similar documents.
# Also include querying by metadata, namely date range.

import os
from dataclasses import dataclass
from datetime import date

import pinecone
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.embeddings.openai import OpenAIEmbeddings

# load environment variables
load_dotenv()

# initialise pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")


@dataclass
class VectorMatch:
    id: str
    ordinal: date
    score: float

    def __init__(self, id: str, ordinal: float, score: float):
        # parse data values
        self.id = id
        self.ordinal = date.fromordinal(int(ordinal))
        self.score = score


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


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
        top_k=3,
        filter=filter,
        include_metadata=True,
    )["matches"]

    # creates VectorMatch objects
    matches = [
        VectorMatch(id=r["id"], ordinal=r["metadata"]["ordinal"], score=r["score"])
        for r in results
    ]

    return {"results": matches}


# test
if __name__ == "__main__":
    # get vector embedding from text query
    query_embedding = embeddings_model.embed_query("China")

    # check if pinecone index exists
    assert os.getenv("PINECONE_INDEX") in pinecone.list_indexes()
    pinecone_index = pinecone.Index(os.getenv("PINECONE_INDEX"))

    # query pinecone index
    results = pinecone_index.query(
        query_embedding,
        top_k=10,
        include_metadata=True,
    )
    print(results)
