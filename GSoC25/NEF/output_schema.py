from pydantic import BaseModel, Field

class DisambiguationResult(BaseModel):
    subject_index: int = Field(description="The integer index of the correct subject from the provided list.")
    predicate_uri: str = Field(description="The exact URI of the predicate selected from the allowed list.")
    object_index: int = Field(description="The integer index of the correct object from the provided list.")