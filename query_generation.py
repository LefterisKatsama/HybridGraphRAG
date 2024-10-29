from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

def generate_full_text_query(input: str) -> str:
    """Generate a full-text search query for a given input string."""
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()
