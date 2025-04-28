import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
import llama_cpp
from llama_cpp import Llama
from outlines import generate, models
from outlines.integrations.utils import convert_json_schema_to_str
from outlines.fsm.json_schema import build_regex_from_schema


class Node(BaseModel):
    """Node of the Knowledge Graph"""

    id: int = Field(..., description="Unique identifier of the node")
    label: str = Field(..., description="Label of the node")
    property: str = Field(..., description="Property of the node")


class Edge(BaseModel):
    """Edge of the Knowledge Graph"""

    source: int = Field(..., description="Unique source of the edge")
    target: int = Field(..., description="Unique target of the edge")
    label: str = Field(..., description="Label of the edge")
    property: str = Field(..., description="Property of the edge")

class KnowledgeGraph(BaseModel):
    """Generated Knowledge Graph"""

    nodes: List[Node] = Field(..., description="List of nodes of the knowledge graph")
    edges: List[Edge] = Field(..., description="List of edges of the knowledge graph")

json_schema = KnowledgeGraph.model_json_schema()
schema_str = convert_json_schema_to_str(json_schema=json_schema)
regex_str = build_regex_from_schema(schema_str)

def generate_hermes_prompt(user_prompt):
    return (
        "<|im_start|>system\n"
        "You are a world class AI model who generates complex knowledge graphs in JSON format. "
        "Create a diverse set of interconnected relationships between multiple entities. "
        "Ensure that relationships are not just centered around one entity, but form a network of connections. "
        f"Here's the json schema you must adhere to:\n<schema>\n{json_schema}\n</schema>\n"
        "Remember to create multiple nodes and establish various relationships between them.<|im_end|>\n"
        "<|im_start|>user\n"
        + user_prompt
        + "\nDo not hallucinate or generate outside the text given."
        + "<|im_end|>"
        + "\n<|im_start|>assistant\n"
        "<schema>"
    )

def prompt_to_json(user_prompt, llama_model):
    prompt = generate_hermes_prompt(user_prompt)
    generator = generate.regex(llama_model, regex_str)
    response = generator(prompt, max_tokens=4096, temperature=0, seed=42)
    return response


def response_to_triples(user_prompt, llama_model):

    response = prompt_to_json(user_prompt, llama_model)
    json_response = json.loads(response)
    nodes = json_response["nodes"]
    edges = json_response['edges']
    node_map = {node['id']: node['label'] for node in nodes}

    triples = []
    for edge in edges:
        source_label = node_map[edge['source']]
        target_label = node_map[edge['target']]
        edge_label = edge['label']
        triples.append((source_label, edge_label, target_label))

    triples_df = pd.DataFrame(triples, columns=['Subject', 'Predicate', 'Object'])

    return triples_df
