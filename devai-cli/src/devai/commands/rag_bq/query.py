import click
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_google_community import BigQueryVectorStore

PROJECT_ID = ""  # @param {type:"string"}
REGION = ""  # @param {type:"string"}
DATASET = ""  # @param {type: "string"}
TABLE = ""  # @param {type: "string"}

@click.command()
@click.option('-q', '--qry', required=False, type=str, default="")
def query(qry):

    # Load the BigQuery DB
    EMBEDDING_QPM = 100
    EMBEDDING_NUM_BATCH = 5
    embeddings = VertexAIEmbeddings(
        requests_per_minute=EMBEDDING_QPM,
        num_instances_per_batch=EMBEDDING_NUM_BATCH,
        model_name="textembedding-gecko@latest",
    )
    db = BigQueryVectorStore(
        project_id=PROJECT_ID,
        dataset_name=DATASET,
        table_name=TABLE,
        location=REGION,
        embedding=embeddings,
    )
    

    # Create a retriever from ChromaDB (adjust parameters as needed)
    # Get top 3 most similar documents
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3})

    # Load the Gemini Pro model
    llm = ChatVertexAI(
        model_name="gemini-1.5-pro",
        safety_settings={},
        temperature=.1,
        # max_output_tokens=256,
        # top_p=0.9,
        # presence_penalty=0.1,
        convert_system_message_to_human=True
    )
    
    # Create a RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True)

    template="Respond to the following query as best you can using the context provided. Keep your answers short and concise. If you don't know say you don't know. {qry}"
    query = template.format(qry=qry)
    result = qa.invoke({"query": query})
    answer = result['result']
    source_documents = result['source_documents']
    print(f"Answer: {answer}")
    print("\nRelevant files:")
    for doc in source_documents:
        print(doc.metadata['source'])
    print("\n")
