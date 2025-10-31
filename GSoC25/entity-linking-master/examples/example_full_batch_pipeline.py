from batch_preprocessing.full_batch_pipeline import full_batch_entity_linking

entity_contexts = [
    {"mention": "Apple", "context": "I work at Apple"},
    {"mention": "Apple", "context": "I eat an apple every day"},
    {"mention": "Paris", "context": "I visited Paris last summer"},
    {"mention": "Barack Obama", "context": "Barack Obama was the 44th president of the United States."},
    {"mention": "Tesla", "context": "Tesla is a leading electric car company."},
    {"mention": "Amazon", "context": "Amazon is the world's largest online retailer."},
    {"mention": "Python", "context": "Python is a popular programming language."},
    {"mention": "Python", "context": "A python is a type of snake."},
    {"mention": "Meta", "context": "Meta Platforms is the parent company of Facebook."},
    {"mention": "Meta", "context": "The concept of meta-analysis is important in statistics."},
    {"mention": "Cambridge", "context": "Cambridge is a university town in England."},
    {"mention": "Cambridge", "context": "Cambridge Analytica was a political consulting firm."}
]

print("\n=== Full Batch Entity Linking Pipeline Output ===")
df = full_batch_entity_linking(entity_contexts, canonical_chunk_size=5, context_chunk_size=5, dbpedia_chunk_size=5)
print(df) 