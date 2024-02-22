**This Github shows you how to integrate Ollama with Atlas vector Search using Langchain to build a RAG App**

The github is linked to this article [here](https://medium.com/@eugenetan_91090/what-is-ollama-dfdaa40cfbca) - check it out for more context.

**Getting started**

_1.Encode vectors to your MongoDB cluster by running_
` python3 encoder.py`

- make sure config.py is declared with the fields (db_name, mongo_uri, coll_name)

```
mongo_uri = "mongodb+srv://user:password@<your-atlas-uri>/?retryWrites=true&w=majority"
db_name = "sample_mflix"
coll_name = "movies"
```

_2. Load your sample atlas data set_

- Follow the instructions [here](https://www.mongodb.com/developer/products/atlas/atlas-sample-datasets/)

_3. Create your Atlas vector search index_

```
{
  "fields": [
    {
      "numDimensions": 384,
      "path": "plot_embedding_hf",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

_4. Run your Ollama | MongoDB RAG app with the command_
<br/>
`streamlit run main.py`


