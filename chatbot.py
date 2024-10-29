from env_setup import *
from neo4j_setup import setup_fulltext_index
from data_ingestion import ingest_data
from vector_search import setup_vector_index
from entity_extraction import get_entity_extraction_chain
from retriever import retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up Neo4j graph
setup_fulltext_index()

# Ingest data into Neo4j
graph = Neo4jGraph()
ingest_data(graph)

# Set up vector index
vector_index = setup_vector_index(graph)

# Entity extraction chain
entity_chain = get_entity_extraction_chain()

# Prompt and Chain for answering questions
template = """Answer the question based on the following context, prioritizing the structured data if available. If structured data is not provided, use the unstructured data. If the provided data are not relevant to the question say that the information is not provided by the context
Context: {context}

Question: {question}
Use natural language and be concise.
Example of question-answer pair:
question: What is the capital of France?
answer: The capital of France is Paris
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')
search_query = RunnableLambda(lambda x: (x["question"], x["rag_mode"]))
chain = (
    RunnableParallel(   
        {
            "context": search_query | (lambda x: retriever(graph, vector_index, entity_chain, x[0], x[1]))
,
            "question": search_query| (lambda x: x[0]),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/messages', methods=['POST'])
def handle_message():
    data = request.json
    question = data.get('question')
    rag_mode = data.get('ragMode')

    print(f"Received question: {question}, rag_mode: {rag_mode}")

    # Query for an answer
    # answer = chain.invoke({"question": question})

    input_data = {
        "question": question,
        "rag_mode": rag_mode
    }
    answer = chain.invoke(input_data)
    
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
