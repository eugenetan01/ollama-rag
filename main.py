import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import config
from langchain_community.llms import Ollama

# Initialize MongoDB client
uri = config.mongo_uri
client = MongoClient(uri)
db_name = "sample_mflix"
collection_name = "movies"
collection = client[db_name][collection_name]

# Initialize text embedding model (encoder)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name = "vector_index"
vector_field_name = "plot_embedding_hf"
text_field_name = "title"

# Specify the MongoDB Atlas database and collection for vector search
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=index_name,
    embedding_key=vector_field_name,
    text_key=text_field_name,
)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Run the LLM (quantized local version by The Bloke's)
llm = Ollama(model="llama2")


# Streamlit App
def main():
    st.title("Movies Retrieval GPT App")

    # User input
    query = st.text_input("Enter your query:")

    # Retrieve context data from MongoDB Atlas Vector Search
    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})  # Modify this line

    # Query LLM with user input and context data
    if st.button("Query LLM"):
        with st.spinner("Querying LLM..."):

            response = llm.invoke(query)

            st.text("Llama2 Response:")
            st.text(response)


if __name__ == "__main__":
    main()
