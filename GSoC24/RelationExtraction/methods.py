
import outlines
from pydantic import BaseModel, Field

# --- 1. Define the Schema (The Shape of Output) ---
class Triple(BaseModel):
    subject: str = Field(..., description="The subject of the triple")
    predicate: str = Field(..., description="The predicate or relation")
    object: str = Field(..., description="The object of the triple")

class TriplesResponse(BaseModel):
    triples: list[Triple]

def get_triples_from_sentence(user_prompt, model):
    """
    Extracts RDF triples from a sentence using Modern Outlines v1.2+ API.
    """
    print(f"   Processing input ({len(user_prompt)} chars)...")

    # --- 2. Prompt Engineering ---
    system_prompt = (
        "You are an expert Knowledge Graph engineer. "
        "Extract structured RDF triples (Subject, Predicate, Object) from the text. "
        "Return the result as a strict JSON object."
    )
    
    # Llama-3 style chat formatting is often helpful, but raw text works too
    full_prompt = f"{system_prompt}\n\nText: {user_prompt}\n\nJSON Output:"

    try:
        # --- 3. Run Inference (The V1.2 Way) ---
        # FIX: We must use 'type' or 'output_type' depending on exact version, 
        # but passing the schema as the second argument is the standard pattern 
        # for the 'generate.json' replacement.
        # However, CodeRabbit suggests using the keyword argument for safety.
        result = model(full_prompt, output_type=TriplesResponse)
        
        # --- 4. Convert to List of Dictionaries ---
        triples_data = []
        for trip in result.triples:
            triples_data.append({
                "sub": trip.subject,
                "rel": trip.predicate,
                "obj": trip.object
            })
            
        print(f"   Found {len(triples_data)} triples.")
        return triples_data

    except Exception as e:
        print(f"   Extraction failed: {e}")
        return []