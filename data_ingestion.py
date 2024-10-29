# from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from update_labels import update_labels


def ingest_data(graph):
    """Ingest data into Neo4j graph, with checks to prevent duplication.""" 
    # Check if the document is already in the graph
    existing_documents = graph.query("MATCH (d:Document) RETURN count(d) AS doc_count")
    if existing_documents[0]['doc_count'] > 0:
        print("Data already ingested, skipping ingestion.")
        return

    # Otherwise, ingest new data
    print("Ingesting new data into the graph...")
    # Load Wikipedia data
    #raw_documents = WikipediaLoader(query="Deadpool & Wolverine").load()
    # Load Word data
    raw_documents = Docx2txtLoader("./Documents/Deadpool.docx").load()


    # Split documents into chunks
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents)

    # Transform documents into a graph format
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Add documents to Neo4j graph
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    update_labels(graph)