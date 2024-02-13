__1.Encode vectors to your MongoDB cluster by running__
``` python3 encoder.py```
- make sure config.py is declared with the fields (db_name, mongo_uri, coll_name)
```
mongo_uri = "mongodb+srv://user:password@<your-atlas-uri>/?retryWrites=true&w=majority"
db_name = "sample_mflix"
coll_name = "movies"
```

__2. Load your sample atlas data set__
- Follow the instructions [here](https://www.mongodb.com/developer/products/atlas/atlas-sample-datasets/)
   
__3. Create your Atlas vector search index__
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

__3. Run your Ollama | MongoDB RAG app with the command__
```streamlit run main.py```
