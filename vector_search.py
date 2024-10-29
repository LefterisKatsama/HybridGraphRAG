from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

def setup_vector_index(graph):
    """Set up Neo4j vector index, ensuring it's created only if it doesn't already exist."""
    
        # Check if the vector index already exists (for Neo4j 4.x and later)
    existing_indexes = graph.query("SHOW INDEXES YIELD name RETURN name")
    
# Check if the 'vector' index already exists
    index_exists = any(index['name'] == 'vector' for index in existing_indexes)
    
    if index_exists:
        print("Vector index already exists, skipping index creation.")
        # Create and return the existing vector index
        vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return vector_index
    # Otherwise, create the vector index
    print("Creating vector index...")
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    print("Vector index created.")
    return vector_index