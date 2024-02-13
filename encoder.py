from sentence_transformers import SentenceTransformer
import pymongo
import config

mongo_uri = config.mongo_uri
db = config.db_name
collection = config.coll_name

# initialize db connection
connection = pymongo.MongoClient(mongo_uri)
collection = connection[db][collection]

# define transofrmer model (from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

for x in collection.find({"plot_embedding_hf": {"$exists": False}}, {}).limit(10):
    # checking if vector already computed for this doc
    if "vector" not in x.keys():
        if "title" in x.keys():
            movieid = x["_id"]
            title = x["title"]
            print("computing vector.. title: " + title)
            text = title
            fullplot = None

            # if fullpplot field present, concat it with title
            if "fullplot" in x.keys():
                fullplot = x["fullplot"]
                text = text + ". " + fullplot

            vector = model.encode(text).tolist()

            collection.update_one(
                {"_id": movieid},
                {
                    "$set": {
                        "plot_embedding_hf": vector,
                        "title": title,
                        "fullplot": fullplot,
                    }
                },
                upsert=True,
            )
            print("vector computed: " + str(x["_id"]))
    else:
        print("vector already computed")
