from env_setup import *
from langchain_community.graphs import Neo4jGraph

def update_labels(graph):
    """
    Update labels in the Neo4j graph by replacing spaces with underscores.

    Parameters:
    - graph: An instance of the Neo4jGraph to execute queries on.
    """
    # Query to get all labels and transform them
    result = graph.query(
        """
        CALL db.labels() YIELD label
        WITH label, replace(label, ' ', '_') AS transformed_label
        RETURN label, transformed_label
        """
    )

    # Iterate through each label, add the new label, and remove the old one
    for record in result:
        old_label = record['label']
        new_label = record['transformed_label']
        
        # Skip if the label hasn't changed (no space to underscore transformation)
        if old_label == new_label:
            continue
        
        # Query to add the new label and remove the old label
        graph.query(
            f"""
            MATCH (n:`{old_label}`)
            CALL apoc.create.addLabels(n, ['{new_label}']) YIELD node AS addedNode
            CALL apoc.create.removeLabels(addedNode, ['{old_label}']) YIELD node AS finalNode
            RETURN finalNode
            """
        )
