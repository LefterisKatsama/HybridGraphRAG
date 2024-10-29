from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List

class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(..., description="All the person, organization, or business entities that appear in the text")

def get_entity_extraction_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ]
    )

    llm = ChatOpenAI(temperature=0)
    entity_chain = prompt | llm.with_structured_output(Entities)
    return entity_chain