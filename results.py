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
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up Neo4j graph
setup_fulltext_index()
graph = Neo4jGraph()

# Set up vector index
vector_index = setup_vector_index(graph)

# Entity extraction chain
entity_chain = get_entity_extraction_chain()

# Prompt and Chain for answering questions
template = """Answer the question based on the following context, using only the structured data if available and enough to answer the question. If structured data is not provided, use only the unstructured data.
context: {context}

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

rag_mode = 'Hybrid' #Hybrid or Classic
data_folder = 'Inception/'
# List of input JSON files
file_names = ['./qa_data/'+data_folder+'Questions-Answers/Count.json', './qa_data/'+data_folder+'Questions-Answers/Generic.json', './qa_data/'+data_folder+'Questions-Answers/MultiHop.json', './qa_data/'+data_folder+'Questions-Answers/YesNo.json']


# Load the tokenizer and the pre-trained deBERTa model fine-tuned on MNLI (for entailment)
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli", use_auth_token="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")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli", use_auth_token="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")

# Create 'results' directory if it doesn't exist
output_dir = "./qa_data/"+data_folder+rag_mode+"1_results"
os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it doesn't exist

# Loop through each file, modify the data, and save the results
for fileName in file_names:
    # Load the JSON file
    with open(fileName, 'r') as file:
        data = json.load(file)
    count = 0
    curr_score = 0
    for item in data:
        if 'question' not in item or 'correct_answer' not in item or 'generated_answer' not in item:
            continue
        count += 1
        question = item['question']
        correct_answer = item['correct_answer']
        item['generated_answer'] = chain.invoke({"question":question, "rag_mode":rag_mode})
        generated_answer = item['generated_answer']
        # Prepare claim and evidence for entailment classification
        claim = f"{question} {generated_answer}"
        evidence = f"{question} {correct_answer}"

        # Tokenize the inputs
        inputs = tokenizer(claim, evidence, return_tensors="pt", truncation=True, padding=True)

        # Perform prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # The logits correspond to the classes: [contradiction, neutral, entailment]
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).tolist()[0]

        # Get the index of the highest probability
        predicted_class = torch.argmax(logits, dim=1).item()

        # Define the label mapping
        labels = ["contradiction", "neutral", "entailment"]
        entailment_prob = probs[2]
        contradiction_prob = probs[0]
        if labels[predicted_class] == "neutral":
            item['score'] = probs[predicted_class]/2
        else:
            item['score'] = entailment_prob
        curr_score += item['score']
    item['final_score'] = curr_score/count

    # Define the output file path (saving it to the 'results' folder)
    base_name = os.path.basename(fileName)  # Extracts the filename
    output_file_name = os.path.join(output_dir, base_name.replace('.json', "_" + rag_mode +"_results.json"))

    # Save the updated data back to a new JSON file in the 'results' directory
    with open(output_file_name, 'w') as output_file:
        json.dump(data, output_file, indent=4)  # Save with pretty-print (indent=4)

    print(f"Results saved to: {output_file_name}")


