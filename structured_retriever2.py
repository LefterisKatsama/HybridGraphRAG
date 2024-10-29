from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from env_setup import *
import requests

def structured_retriever2(graph, question):
    # Initialize the Neo4j graph and refresh its schema
    graph.refresh_schema()
    
    # Create the GraphCypherQAChain with the LLM and the graph
    cypher_chain = GraphCypherQAChain.from_llm(
        cypher_llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini'),
        qa_llm = ChatOpenAI(temperature=0), graph=graph, verbose=True, return_direct=True
    )
    
    # Run the query using the question parameter and return the result
    output = cypher_chain.run(question)
    return output