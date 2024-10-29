from structured_retriever import structured_retriever
from structured_retriever2 import structured_retriever2

def retriever(graph, vector_index, entity_chain, question: str, rag_mode: str) -> str:
    final_data = ""
    if rag_mode == 'Hybrid':
        #structured_data = structured_retriever(graph, entity_chain, question)
        structured_data = structured_retriever2(graph, question)

        if structured_data:
            print("Entity matched, structured retriever used")
            final_data += f"""Structured data:
    {structured_data}
    """
        else:
            print("No entity matched, structured retriever failed")

    # Whether or not structured data was found, add unstructured data
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question, 3)]
    if unstructured_data:
        final_data += f"""\nUnstructured data:
{"#Document ".join(unstructured_data)}
"""
    else:
        final_data += "\nNo unstructured data found."
    print(final_data)
    return final_data
