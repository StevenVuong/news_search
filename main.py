import os
from pathlib import Path

import pandas as pd
import pinecone
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# initialise pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)


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

        print(df.head())

        break

    # convert xlsx to csv file

    # read csv files from csv_dir
