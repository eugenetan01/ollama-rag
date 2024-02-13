__This Github shows you how to integrate Ollama with Atlas vector Search using Langchain to build a RAG App__

The github is linked to this article [here](https://medium.com/@eugenetan_91090/what-is-ollama-dfdaa40cfbca) - check it out for more context.

__Getting started__

*1.Encode vectors to your MongoDB cluster by running*
``` python3 encoder.py```
- make sure config.py is declared with the fields (db_name, mongo_uri, coll_name)
```
mongo_uri = "mongodb+srv://user:password@<your-atlas-uri>/?retryWrites=true&w=majority"
db_name = "sample_mflix"
coll_name = "movies"
```

*2. Load your sample atlas data set*
- Follow the instructions [here](https://www.mongodb.com/developer/products/atlas/atlas-sample-datasets/)
   
*3. Create your Atlas vector search index*
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

*4. Run your Ollama | MongoDB RAG app with the command*
<br/>
```streamlit run main.py```
