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
from langchain.prompts import PromptTemplate

# Initialize MongoDB client
uri = config.mongo_uri
client = MongoClient(uri)
db_name = config.db_name
coll_name = config.coll_name
collection = client[db_name][coll_name]

# Initialize text embedding model (encoder)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name = "vector_ollama"
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

# vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Run the LLM (quantized local version by The Bloke's)
llm = Ollama(model="llama2", callback_manager=callback_manager)


# Streamlit App
def main():
    st.title("Movies Retrieval GPT App")

    # User input
    query = st.text_input("Enter your query:")

    retriever = vectorStore.as_retriever()
    # Modify this line

    prompt_template = """
    <s>[INST] <<SYS>>
    Use the following context to answer the question at the end as the only source of truth. The following contain movie titles along with the associated movie description. 
    Just share the movie title and description as it is provided in the context.
    The movie titles and their corresponding plots are as follow:
    <</SYS>>
    {context}

    Question: {question}[/INST]
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Query LLM with user input and context data
    if st.button("Query LLM"):
        with st.spinner("Querying LLM..."):
            qa = RetrievalQA.from_chain_type(
                llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT},
            )
            # retriever_output = qa.run(query)
            retriever_output = qa(query)
            # response = qa({"query": query})

            st.text("Llama2 Response:")
            st.text(retriever_output)


if __name__ == "__main__":
    main()
