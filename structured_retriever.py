from query_generation import generate_full_text_query

def structured_retriever(graph, entity_chain, question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    print('entities:',entities)
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    print(result)
    return result
