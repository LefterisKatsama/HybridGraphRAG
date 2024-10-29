from langchain_community.graphs import Neo4jGraph

# Initialize Neo4j Graph
graph = Neo4jGraph()

# Create a full-text index for entities
def setup_fulltext_index():
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
    )
