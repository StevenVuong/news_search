# News Search

To start app:

```bash
poetry run streamlit run app.py
```

## Add Data to Vector Database

```sh
poetry run python -m establish_index.py
```

## Deploy to Azure Web Services

Create a `requirements.txt` file:

```sh
poetry export --without-hashes --format=requirements.txt > requirements.txt
```

Create web app in Azure

```sh
az webapp up \
  --resource-group rg-openai-demo-steve \
  --os-type Linux \
  --name news-streamlit-search \
  --runtime PYTHON:3.10 \
  --sku S2
```

Add to:
`python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0`
And .env variables also.
